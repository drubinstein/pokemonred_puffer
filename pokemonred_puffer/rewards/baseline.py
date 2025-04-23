import numpy as np
from omegaconf import DictConfig, OmegaConf

from pokemonred_puffer.data.events import EVENTS, REQUIRED_EVENTS
from pokemonred_puffer.data.items import REQUIRED_ITEMS, USEFUL_ITEMS
from pokemonred_puffer.data.tilesets import Tilesets
from pokemonred_puffer.environment import RedGymEnv


MUSEUM_TICKET = (0xD754, 0)


class BaselineRewardEnv(RedGymEnv):
    def __init__(self, env_config: DictConfig, reward_config: DictConfig):
        super().__init__(env_config)
        self.reward_config = OmegaConf.to_object(reward_config)
        self.max_event_rew = 0

    # TODO: make the reward weights configurable
    def get_game_state_reward(self):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm

        return {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.0000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.0000010,
            "obtained_move_ids": sum(self.obtained_move_ids) * 0.00010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            # "level": self.get_levels_reward(),
            # "opponent_level": self.max_opponent_level,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5,
            # "heal": self.total_healing_rew,
            "explore": sum(sum(tileset.values()) for tileset in self.seen_coords.values()) * 0.012,
            # "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_hm(0xF)),
            "cut_coords": sum(self.cut_coords.values()) * 1.0,
            "cut_tiles": sum(self.cut_tiles.values()) * 1.0,
            "met_bill": 5 * int(self.events.get_event("EVENT_MET_BILL")),
            "used_cell_separator_on_bill": 5
            * int(self.events.get_event("EVENT_USED_CELL_SEPARATOR_ON_BILL")),
            "ss_ticket": 5 * int(self.events.get_event("EVENT_GOT_SS_TICKET")),
            "met_bill_2": 5 * int(self.events.get_event("EVENT_MET_BILL_2")),
            "bill_said_use_cell_separator": 5
            * int(self.events.get_event("EVENT_BILL_SAID_USE_CELL_SEPARATOR")),
            "left_bills_house_after_helping": 5
            * int(self.events.get_event("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING")),
            "got_hm01": 5 * int(self.events.get_event("EVENT_GOT_HM01")),
            "rubbed_captains_back": 5 * int(self.events.get_event("EVENT_RUBBED_CAPTAINS_BACK")),
            "start_menu": self.seen_start_menu * 0.01,
            "pokemon_menu": self.seen_pokemon_menu * 0.1,
            "stats_menu": self.seen_stats_menu * 0.1,
            "bag_menu": self.seen_bag_menu * 0.1,
            "action_bag_menu": self.seen_action_bag_menu * 0.1,
            # "blackout_check": self.blackout_check * 0.001,
            "rival3": self.reward_config["event"] * int(self.read_m("wSSAnne2FCurScript") == 4),
        }

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            np.sum(self.events.get_events(EVENTS))
            - self.base_event_flags
            - int(
                self.events.get_event("EVENT_BOUGHT_MUSEUM_TICKET"),
            ),
            0,
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class TeachCutReplicationEnv(BaselineRewardEnv):
    def get_game_state_reward(self):
        return {
            "event": self.reward_config["event"] * self.update_max_event_rew(),
            "met_bill": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL")),
            "used_cell_separator_on_bill": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_USED_CELL_SEPARATOR_ON_BILL")),
            "ss_ticket": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_GOT_SS_TICKET")),
            "met_bill_2": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL_2")),
            "bill_said_use_cell_separator": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_BILL_SAID_USE_CELL_SEPARATOR")),
            "left_bills_house_after_helping": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING")),
            "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
            "obtained_move_ids": self.reward_config["obtained_move_ids"]
            * sum(self.obtained_move_ids),
            "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
            "level": self.reward_config["level"] * self.get_levels_reward(),
            "badges": self.reward_config["badges"] * self.get_badges(),
            "exploration": self.reward_config["exploration"]
            * sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
            "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
            "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
            "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
            "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
            "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
            "rival3": self.reward_config["event"] * int(self.read_m("wSSAnne2FCurScript") == 4),
        }


class TeachCutReplicationEnvFork(BaselineRewardEnv):
    def get_game_state_reward(self):
        return {
            "event": self.reward_config["event"] * self.update_max_event_rew(),
            "met_bill": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL")),
            "used_cell_separator_on_bill": (
                self.reward_config["bill_saved"]
                * int(self.events.get_event("EVENT_USED_CELL_SEPARATOR_ON_BILL"))
            ),
            "ss_ticket": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_USED_CELL_SEPARATOR_ON_BILL")),
            "met_bill_2": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL_2")),
            "bill_said_use_cell_separator": (
                self.reward_config["bill_saved"]
                * int(self.events.get_event("EVENT_BILL_SAID_USE_CELL_SEPARATOR"))
            ),
            "left_bills_house_after_helping": (
                self.reward_config["bill_saved"]
                * int(self.events.get_event("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING"))
            ),
            "obtained_move_ids": self.reward_config["obtained_move_ids"]
            * sum(self.obtained_move_ids),
            "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
            "badges": self.reward_config["badges"] * self.get_badges(),
            "exploration": self.reward_config["exploration"]
            * sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
            "explore_npcs": self.reward_config["explore_npcs"] * sum(self.seen_npcs.values()),
            "explore_hidden_objs": (
                self.reward_config["explore_hidden_objs"] * sum(self.seen_hidden_objs.values())
            ),
            "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles),
            "start_menu": (
                self.reward_config["start_menu"] * self.seen_start_menu * int(self.taught_cut)
            ),
            "pokemon_menu": (
                self.reward_config["pokemon_menu"] * self.seen_pokemon_menu * int(self.taught_cut)
            ),
            "stats_menu": (
                self.reward_config["stats_menu"] * self.seen_stats_menu * int(self.taught_cut)
            ),
            "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu * int(self.taught_cut),
            "taught_cut": self.reward_config["taught_cut"] * int(self.taught_cut),
            "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
            "level": self.reward_config["level"] * self.get_levels_reward(),
        }

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class CutWithObjectRewardsEnv(BaselineRewardEnv):
    def get_game_state_reward(self):
        return {
            "event": self.reward_config["event"] * self.update_max_event_rew(),
            "met_bill": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL")),
            "used_cell_separator_on_bill": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING")),
            "ss_ticket": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_GOT_SS_TICKET")),
            "met_bill_2": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_MET_BILL_2")),
            "bill_said_use_cell_separator": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_BILL_SAID_USE_CELL_SEPARATOR")),
            "left_bills_house_after_helping": self.reward_config["bill_saved"]
            * int(self.events.get_event("EVENT_LEFT_BILLS_HOUSE_AFTER_HELPING")),
            "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
            "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
            "obtained_move_ids": self.reward_config["obtained_move_ids"]
            * sum(self.obtained_move_ids),
            "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
            "level": self.reward_config["level"] * self.get_levels_reward(),
            "badges": self.reward_config["badges"] * self.get_badges(),
            "exploration": self.reward_config["exploration"]
            * sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
            "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
            "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
            "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
            "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
            "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
            "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
            "rival3": self.reward_config["event"] * int(self.read_m("wSSAnne2FCurScript") == 4),
            "rocket_hideout_found": self.reward_config["rocket_hideout_found"]
            * int(self.events.get_event("EVENT_FOUND_ROCKET_HIDEOUT")),
            "explore_hidden_objs": sum(self.seen_hidden_objs.values())
            * self.reward_config["explore_hidden_objs"],
            "seen_action_bag_menu": self.seen_action_bag_menu
            * self.reward_config["seen_action_bag_menu"],
        }

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class CutWithObjectRewardRequiredEventsEnv(BaselineRewardEnv):
    def get_game_state_reward(self):
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        bag = np.array(self.pyboy.memory[wBagItems : wBagItems + 40], dtype=np.uint8)
        numBagItems = self.read_m("wNumBagItems")
        # item ids start at 1 so using 0 as the nothing value is okay
        bag[2 * numBagItems :] = 0
        bag_item_ids = bag[::2]

        return (
            {
                "event": self.reward_config["event"] * self.update_max_event_rew(),
                "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
                "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
                "obtained_move_ids": self.reward_config["obtained_move_ids"]
                * sum(self.obtained_move_ids),
                "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
                "level": self.reward_config["level"] * self.get_levels_reward(),
                "badges": self.reward_config["badges"] * self.get_badges(),
                "exploration": self.reward_config["exploration"]
                * sum(sum(tileset.values()) for tileset in self.seen_coords.values()),
                "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
                "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
                "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
                "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
                "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
                "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
                "explore_hidden_objs": sum(self.seen_hidden_objs.values())
                * self.reward_config["explore_hidden_objs"],
                "seen_action_bag_menu": self.seen_action_bag_menu
                * self.reward_config["seen_action_bag_menu"],
                "pokecenter_heal": self.pokecenter_heal * self.reward_config["pokecenter_heal"],
                "rival3": self.reward_config["required_event"]
                * int(self.read_m("wSSAnne2FCurScript") == 4),
                "game_corner_rocket": self.reward_config["required_event"]
                * float(self.missables.get_missable("HS_GAME_CORNER_ROCKET")),
                "saffron_guard": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK")),
                "lapras": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GOT_LAPRAS")),
            }
            | {
                event: self.reward_config["required_event"] * float(self.events.get_event(event))
                for event in REQUIRED_EVENTS
            }
            | {
                item.name: self.reward_config["required_item"] * float(item.value in bag_item_ids)
                for item in REQUIRED_ITEMS
            }
            | {
                item.name: self.reward_config["useful_item"] * float(item.value in bag_item_ids)
                for item in USEFUL_ITEMS
            }
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class ObjectRewardRequiredEventsEnvTilesetExploration(BaselineRewardEnv):
    def get_game_state_reward(self):
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        numBagItems = self.read_m("wNumBagItems")
        bag_item_ids = set(self.pyboy.memory[wBagItems : wBagItems + 2 * numBagItems : 2])

        return (
            {
                "event": self.reward_config["event"] * self.update_max_event_rew(),
                "seen_pokemon": self.reward_config["seen_pokemon"] * sum(self.seen_pokemon),
                "caught_pokemon": self.reward_config["caught_pokemon"] * sum(self.caught_pokemon),
                "obtained_move_ids": self.reward_config["obtained_move_ids"]
                * sum(self.obtained_move_ids),
                "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
                "level": self.reward_config["level"] * self.get_levels_reward(),
                "badges": self.reward_config["badges"] * self.get_badges(),
                "cut_coords": self.reward_config["cut_coords"] * sum(self.cut_coords.values()),
                "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
                "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
                "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
                "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
                "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
                "explore_hidden_objs": sum(self.seen_hidden_objs.values()),
                "explore_signs": sum(self.seen_signs.values())
                * self.reward_config["explore_signs"],
                "seen_action_bag_menu": self.seen_action_bag_menu
                * self.reward_config["seen_action_bag_menu"],
                "pokecenter_heal": self.pokecenter_heal * self.reward_config["pokecenter_heal"],
                "rival3": self.reward_config["required_event"]
                * int(self.read_m("wSSAnne2FCurScript") == 4),
                "game_corner_rocket": self.reward_config["required_event"]
                * float(self.missables.get_missable("HS_GAME_CORNER_ROCKET")),
                "saffron_guard": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK")),
                "lapras": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GOT_LAPRAS")),
                "a_press": len(self.a_press) * self.reward_config["a_press"],
                "warps": len(self.seen_warps) * self.reward_config["explore_warps"],
                "use_surf": self.reward_config["use_surf"] * self.use_surf,
            }
            | {
                f"exploration_{tileset.name.lower()}": self.reward_config.get(
                    f"exploration_{tileset.name.lower()}", self.reward_config["exploration"]
                )
                * sum(self.seen_coords.get(tileset.value, {}).values())
                for tileset in Tilesets
            }
            | {
                event: self.reward_config["required_event"] * float(self.events.get_event(event))
                for event in REQUIRED_EVENTS
            }
            | {
                item.name: self.reward_config["required_item"] * float(item.value in bag_item_ids)
                for item in REQUIRED_ITEMS
            }
            | {
                item.name: self.reward_config["useful_item"] * float(item.value in bag_item_ids)
                for item in USEFUL_ITEMS
            }
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class ObjectRewardRequiredEventsMapIds(BaselineRewardEnv):
    def get_game_state_reward(self) -> dict[str, float]:
        _, wBagItems = self.pyboy.symbol_lookup("wBagItems")
        numBagItems = self.read_m("wNumBagItems")
        bag_item_ids = set(self.pyboy.memory[wBagItems : wBagItems + 2 * numBagItems : 2])

        return (
            {
                "cut_tiles": self.reward_config["cut_tiles"] * sum(self.cut_tiles.values()),
                "event": self.reward_config["event"] * self.update_max_event_rew(),
                "seen_pokemon": self.reward_config["seen_pokemon"] * np.sum(self.seen_pokemon),
                "caught_pokemon": self.reward_config["caught_pokemon"]
                * np.sum(self.caught_pokemon),
                "obtained_move_ids": self.reward_config["obtained_move_ids"]
                * np.sum(self.obtained_move_ids),
                "hm_count": self.reward_config["hm_count"] * self.get_hm_count(),
                "level": self.reward_config["level"] * self.get_levels_reward(),
                "badges": self.reward_config["badges"] * self.get_badges(),
                "valid_cut_coords": self.reward_config["valid_cut_coords"]
                * len(self.valid_cut_coords.values()),
                "invalid_cut_coords": self.reward_config["invalid_cut_coords"]
                * len(self.invalid_cut_coords.values()),
                "start_menu": self.reward_config["start_menu"] * self.seen_start_menu,
                "pokemon_menu": self.reward_config["pokemon_menu"] * self.seen_pokemon_menu,
                "stats_menu": self.reward_config["stats_menu"] * self.seen_stats_menu,
                "bag_menu": self.reward_config["bag_menu"] * self.seen_bag_menu,
                "explore_hidden_objs": sum(self.seen_hidden_objs.values()),
                "explore_signs": sum(self.seen_signs.values())
                * self.reward_config["explore_signs"],
                "seen_action_bag_menu": self.seen_action_bag_menu
                * self.reward_config["seen_action_bag_menu"],
                "pokecenter_heal": self.pokecenter_heal * self.reward_config["pokecenter_heal"],
                "rival3": self.reward_config["required_event"]
                * int(self.read_m("wSSAnne2FCurScript") == 4),
                "game_corner_rocket": self.reward_config["required_event"]
                * float(self.missables.get_missable("HS_GAME_CORNER_ROCKET")),
                "saffron_guard": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GAVE_SAFFRON_GUARDS_DRINK")),
                "lapras": self.reward_config["required_event"]
                * float(self.flags.get_bit("BIT_GOT_LAPRAS")),
                "a_press": len(self.a_press) * self.reward_config["a_press"],
                "warps": len(self.seen_warps) * self.reward_config["explore_warps"],
                "use_surf": self.reward_config["use_surf"] * self.use_surf,
                "exploration": self.reward_config["exploration"] * np.sum(self.reward_explore_map),
                "safari_zone": sum(
                    self.reward_config["safari_zone"] * v for k, v in self.safari_zone_steps.items()
                )
                / 502.0,
                "use_ball_count": (
                    0
                    if self.flags.get_bit("BIT_GOT_LAPRAS")
                    else self.reward_config["use_ball_count"]
                )
                * self.use_ball_count,
                "pokeflute_tiles": self.reward_config["pokeflute_tiles"]
                * sum(self.pokeflute_tiles.values()),
                "surf_tiles": self.reward_config["surf_tiles"] * sum(self.surf_tiles.values()),
            }
            | {
                event: self.reward_config["required_event"] * float(self.events.get_event(event))
                for event in REQUIRED_EVENTS
            }
            | {
                item.name: self.reward_config["required_item"] * float(item.value in bag_item_ids)
                for item in REQUIRED_ITEMS
            }
            | {
                item.name: self.reward_config["useful_item"] * float(item.value in bag_item_ids)
                for item in USEFUL_ITEMS
            }
        )

    def get_levels_reward(self):
        party_size = self.read_m("wPartyCount")
        party_levels = [self.read_m(f"wPartyMon{i+1}Level") for i in range(party_size)]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 15:
            return self.max_level_sum
        else:
            return 15 + (self.max_level_sum - 15) / 4


class ObjectRewardRequiredEventsMapIdsFieldMoves(ObjectRewardRequiredEventsMapIds):
    def get_game_state_reward(self) -> dict[str, float]:
        return super().get_game_state_reward() | {
            "valid_pokeflute_coords": self.reward_config["valid_pokeflute_coords"]
            * len(self.valid_pokeflute_coords.values()),
            "invalid_pokeflute_coords": self.reward_config["invalid_pokeflute_coords"]
            * len(self.invalid_pokeflute_coords.values()),
            "valid_surf_coords": self.reward_config["valid_surf_coords"]
            * len(self.valid_surf_coords.values()),
            "invalid_surf_coords": self.reward_config["invalid_surf_coords"]
            * len(self.invalid_surf_coords.values()),
        }
