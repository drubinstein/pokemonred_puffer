import pufferlib.emulation
import pufferlib.models
import pufferlib.pytorch
import torch
from torch import nn

from pokemonred_puffer.data.events import EVENTS_IDXS
from pokemonred_puffer.data.items import Items
from pokemonred_puffer.environment import PIXEL_VALUES


# Because torch.nn.functional.one_hot cannot be traced by torch as of 2.2.0
def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])).to(
        torch.int64
    )


class MultiConvolutionalRNN(pufferlib.models.LSTMWrapper):
    pass


# We dont inherit from the pufferlib convolutional beause we wont be able
# to easily call its __init__ due to our usage of lazy layers
# All that really means is a slightly different forward
class MultiConvolutionalPolicy(nn.Module):
    def __init__(
        self,
        env: pufferlib.emulation.GymnasiumPufferEnv,
        hidden_size: int = 512,
        channels_last: bool = True,
        downsample: int = 1,
    ):
        super().__init__()
        self.is_continuous = False
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
        self.num_actions = env.single_action_space.n
        self.channels_last = channels_last
        self.downsample = downsample
        self.screen_network = nn.Sequential(
            nn.Conv2d(2, 32, 8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # if channels_last:
        #     self.screen_network = self.screen_network.to(memory_format=torch.channels_last)

        self.encode_linear = nn.Sequential(
            nn.Linear(4225, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, self.num_actions)
        self.value_fn = nn.Linear(hidden_size, 1)

        self.two_bit = env.unwrapped.env.two_bit
        self.skip_safari_zone = env.unwrapped.env.skip_safari_zone
        self.use_global_map = env.unwrapped.env.use_global_map

        if self.use_global_map:
            self.global_map_network = nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, 480),
                nn.ReLU(),
            )
            # if channels_last:
            #     self.global_map_network = self.global_map_network.to(
            #         memory_format=torch.channels_last
            #    )

        self.register_buffer(
            "screen_buckets", torch.tensor(PIXEL_VALUES, dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "linear_buckets", torch.tensor([0, 64, 128, 255], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_mask",
            torch.tensor([0xC0, 0x30, 0x0C, 0x03], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_shift", torch.tensor([6, 4, 2, 0], dtype=torch.uint8), persistent=False
        )
        self.register_buffer(
            "unpack_bytes_mask",
            torch.tensor([0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1], dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "unpack_bytes_shift",
            torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8),
            persistent=False,
        )
        # self.register_buffer("badge_buffer", torch.arange(8) + 1, persistent=False)

        # pokemon has 0xF7 map ids
        # Lets start with 4 dims for now. Could try 8
        self.map_embeddings = nn.Embedding(0xFF, 4, dtype=torch.float32)
        # N.B. This is an overestimate
        item_count = max(Items._value2member_map_.keys())
        self.item_embeddings = nn.Embedding(
            item_count, int(item_count**0.25 + 1), dtype=torch.float32
        )

        # Party layers
        self.party_network = nn.Sequential(nn.Linear(40, 6), nn.ReLU(), nn.Flatten())
        self.species_embeddings = nn.Embedding(0xBE, int(0xBE**0.25) + 1, dtype=torch.float32)
        self.type_embeddings = nn.Embedding(0x1A, int(0x1A**0.25) + 1, dtype=torch.float32)
        self.moves_embeddings = nn.Embedding(0xA4, int(0xA4**0.25) + 1, dtype=torch.float32)

        # event embeddings
        n_events = env.env.observation_space["events"].shape[0]
        self.event_embeddings = nn.Embedding(n_events, int(n_events**0.25) + 1, dtype=torch.float32)

    def forward(self, observations):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def encode_observations(self, observations, state=None):
        observations = observations.type(torch.uint8)  # Undo bad cleanrl cast
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        screen = observations["screen"]
        visited_mask = observations["visited_mask"]
        restored_shape = (screen.shape[0], screen.shape[1], screen.shape[2] * 4, screen.shape[3])
        if self.use_global_map:
            global_map = observations["global_map"]
            restored_global_map_shape = (
                global_map.shape[0],
                global_map.shape[1],
                global_map.shape[2] * 4,
                global_map.shape[3],
            )

        if self.two_bit:
            screen = torch.index_select(
                self.screen_buckets,
                0,
                ((screen.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift).flatten().int(),
            ).reshape(restored_shape)
            visited_mask = torch.index_select(
                self.linear_buckets,
                0,
                ((visited_mask.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                .flatten()
                .int(),
            ).reshape(restored_shape)
            if self.use_global_map:
                global_map = torch.index_select(
                    self.linear_buckets,
                    0,
                    ((global_map.reshape((-1, 1)) & self.unpack_mask) >> self.unpack_shift)
                    .flatten()
                    .int(),
                ).reshape(restored_global_map_shape)
        # badges = self.badge_buffer <= observations["badges"]
        map_id = self.map_embeddings(observations["map_id"].int()).squeeze(1)
        blackout_map_id = self.map_embeddings(observations["blackout_map_id"].int()).squeeze(1)
        # The bag quantity can be a value between 1 and 99
        # TODO: Should items be positionally encoded? I dont think it matters
        items = (
            self.item_embeddings(observations["bag_items"].int())
            * (observations["bag_quantity"].float().unsqueeze(-1) / 100.0)
        ).squeeze(1)

        # image_observation = torch.cat((screen, visited_mask, global_map), dim=-1)
        image_observation = torch.cat((screen, visited_mask), dim=-1)
        if self.channels_last:
            image_observation = image_observation.permute(0, 3, 1, 2)
            # image_observation = image_observation.to( memory_format=torch.channels_last)
            if self.use_global_map:
                global_map = global_map.permute(0, 3, 1, 2)
                # global_map = global_map.to(memory_format=torch.channels_last)
        if self.downsample > 1:
            image_observation = image_observation[:, :, :: self.downsample, :: self.downsample]

        # party network
        species = self.species_embeddings(observations["species"].int()).float().squeeze(1)
        status = one_hot(observations["status"].int(), 7).float().squeeze(1)
        type1 = self.type_embeddings(observations["type1"].int()).squeeze(1)
        type2 = self.type_embeddings(observations["type2"].int()).squeeze(1)
        moves = (
            self.moves_embeddings(observations["moves"].int())
            .squeeze(1)
            .float()
            .reshape((-1, 6, 4 * self.moves_embeddings.embedding_dim))
        )
        party_obs = torch.cat(
            (
                species,
                observations["hp"].float().unsqueeze(-1) / 714.0,
                status,
                type1,
                type2,
                observations["level"].float().unsqueeze(-1) / 100.0,
                observations["maxHP"].float().unsqueeze(-1) / 714.0,
                observations["attack"].float().unsqueeze(-1) / 714.0,
                observations["defense"].float().unsqueeze(-1) / 714.0,
                observations["speed"].float().unsqueeze(-1) / 714.0,
                observations["special"].float().unsqueeze(-1) / 714.0,
                moves,
            ),
            dim=-1,
        )
        party_latent = self.party_network(party_obs)

        # event_obs = (
        #     observations["events"].float() @ self.event_embeddings.weight
        # ) / self.event_embeddings.weight.shape[0]
        events_obs = (
            (
                (
                    (observations["events"].reshape((-1, 1)) & self.unpack_bytes_mask)
                    >> self.unpack_bytes_shift
                )
                .flatten()
                .reshape((observations["events"].shape[0], -1))[:, EVENTS_IDXS]
            )
            .float()
            .squeeze(1)
        )

        cat_obs = torch.cat(
            (
                self.screen_network(image_observation.float() / 255.0).squeeze(1),
                one_hot(observations["direction"].int(), 4).float().squeeze(1),
                # one_hot(observations["reset_map_id"].int(), 0xF7).float().squeeze(1),
                one_hot(observations["battle_type"].int(), 4).float().squeeze(1),
                # observations["cut_event"].float(),
                # observations["x"].float(),
                # observations["y"].float(),
                # one_hot(observations["map_id"].int(), 0xF7).float().squeeze(1),
                # badges.float().squeeze(1),
                map_id.squeeze(1),
                blackout_map_id.squeeze(1),
                items.flatten(start_dim=1),
                party_latent,
                events_obs,
                observations["rival_3"].float(),
                observations["game_corner_rocket"].float(),
                observations["saffron_guard"].float(),
                observations["lapras"].float(),
            )
            + (() if self.skip_safari_zone else (observations["safari_steps"].float() / 502.0,))
            + (
                (self.global_map_network(global_map.float() / 255.0).squeeze(1),)
                if self.use_global_map
                else ()
            ),
            dim=-1,
        )
        return self.encode_linear(cat_obs)

    def decode_actions(self, flat_hidden, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
