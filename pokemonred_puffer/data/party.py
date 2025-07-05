from ctypes import (
    BigEndianStructure,
    Union,
    c_uint16,
    c_uint8,
    sizeof,
)

from pyboy import PyBoy

from pokemonred_puffer.data.species import Species


class BoxStruct(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("Species", c_uint8),
        ("HP", c_uint16),
        ("BoxLevel", c_uint8),
        ("Status", c_uint8),
        ("Type1", c_uint8),
        ("Type2", c_uint8),
        ("CatchRate", c_uint8),
        ("Moves", 4 * c_uint8),
        ("OTID", c_uint16),
        ("Exp", 3 * c_uint8),
        ("HPExp", c_uint16),
        ("AttackExp", c_uint16),
        ("DefenseExp", c_uint16),
        ("SpeedExp", c_uint16),
        ("SpecialExp", c_uint16),
        ("DVs", 2 * c_uint8),
        ("PP", 4 * c_uint8),
    ]


class PartyStruct(BigEndianStructure):
    _pack_ = 1
    _fields_ = BoxStruct._fields_ + [  # type: ignore
        ("Level", c_uint8),
        ("MaxHP", c_uint16),
        ("Attack", c_uint16),
        ("Defense", c_uint16),
        ("Speed", c_uint16),
        ("Special", c_uint16),
    ]


PARTY_LENGTH_BYTES = 6 * sizeof(PartyStruct)


class PartyMons(Union):
    _fields_ = [("party", 6 * PartyStruct), ("asbytes", c_uint8 * PARTY_LENGTH_BYTES)]

    def __init__(self, emu: PyBoy):
        _, wPartyMons = emu.symbol_lookup("wPartyMons")
        _, wPartyCount = emu.symbol_lookup("wPartyCount")
        self.party_size = emu.memory[wPartyCount]
        self.asbytes = (c_uint8 * PARTY_LENGTH_BYTES)(
            *emu.memory[wPartyMons : wPartyMons + PARTY_LENGTH_BYTES]
        )

    def __getitem__(self, idx):
        return self.party[idx]

    def __repr__(self):
        return str([Species(x.Species).name for x in self.party[: self.party_size]])
