"""Module to handle the subarray configurations"""

import fnmatch
import hashlib
import json
import logging
import operator
import os
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import TypeVar

import numpy as np
from astropy.io import ascii
from marshmallow import Schema, fields, post_load

from skasz.senscalc.utilities import STATIC_DATA_PATH, Telescope

logger = logging.getLogger("senscalc")

SUBARRAY_STORAGE_PATH = STATIC_DATA_PATH / "subarrays"


class MIDArrayConfiguration(Enum):
    """
    Enumeration of SKA MID subarray configurations
    """

    MID_AA05_ALL = "AA0.5"
    MID_AA1_ALL = "AA1"
    MID_AA2_ALL = "AA2"
    MID_AASTAR_ALL = "AA*"
    MID_AASTAR_SKA_ONLY = "AA* (15-m antennas only)"
    MID_AA4_ALL = "AA4"
    MID_AA4_MEERKAT_ONLY = "AA*/AA4 (13.5-m antennas only)"
    MID_AA4_SKA_ONLY = "AA4 (15-m antennas only)"


class LOWArrayConfiguration(Enum):
    """
    Enumeration of SKA LOW subarray configurations.

    Enumerations MUST be case-insensitive and unique as HTTP query parameters
    will be converted to enum instances using str.upper().
    """

    LOW_AA05_ALL = "AA0.5"
    LOW_AA1_ALL = "AA1"
    LOW_AA2_ALL = "AA2"
    LOW_AA2_CORE_ONLY = "AA2 (core only)"
    LOW_AASTAR_ALL = "AA*"
    LOW_AASTAR_CORE_ONLY = "AA* (core only)"
    LOW_AA4_ALL = "AA4"
    LOW_AA4_CORE_ONLY = "AA4 (core only)"
    LOW_CUSTOM = "Custom"


class Subarray(ABC):
    def __init__(self, name: str, label: str, ids: list, telescope: Telescope):
        self.name = name
        self.label = label
        self.ids = ids
        self.telescope = telescope


# Type hint for saying 'a Subarray or a subclass of Subarray')
U = TypeVar("U", bound=Subarray)


class MidSubarray(Subarray):
    def __init__(
        self,
        name: str,
        label: str,
        configuration: str,
        ids: list,
        md5_checksum: str,
    ):
        super().__init__(name, label, ids, telescope=Telescope.MID)
        self.configuration = configuration
        self.md5_checksum = md5_checksum

        # verify the checksum of the configuration file matches

        with open(configuration, "rb") as f:
            content = f.read()
            hashlib.md5().update(content)
            digest = hashlib.md5().hexdigest()
            if digest != md5_checksum:
                raise ValueError(
                    f"Subarray checksum {md5_checksum} does not match"
                    f" configuration {digest}"
                )

        # read configuration

        cols = ["X", "Y", "Z", "Diam", "Station"]

        df = ascii.read(
            os.path.expandvars(configuration),
            comment="#",
            names=cols,
        )

        df.add_column(np.arange(len(df)), name="ID", index=0)
        cols.insert(0, "ID")

        # adding in indexing as this does not work by default with astropy
        df.add_index(cols)

        # select subarray

        df_subarray = df.iloc[ids]

        # how many antennas are SKA or Meerkat?
        self.n_ska = len(fnmatch.filter(df_subarray["Station"], "SKA*"))
        self.n_meer = len(fnmatch.filter(df_subarray["Station"], "M*"))

    def __eq__(self, other):
        """Equality method. The configuration and attribute is ignored here because
        the same file could have been loaded via a different path. The checksum
        is the real test.
        """
        if not isinstance(other, MidSubarray):
            return False

        return (
            self.name,
            self.ids,
            self.md5_checksum,
            self.n_meer,
            self.n_ska,
        ) == (
            other.name,
            other.ids,
            other.md5_checksum,
            other.n_meer,
            other.n_ska,
        )

    def __repr__(self):
        return (
            f"<MidSubarray(name='{self.name}', label='{self.label}',"
            f" n_meer={self.n_meer}, n_ska={self.n_ska})>"
        )


class LowSubarray(Subarray):
    def __init__(self, name: str, label: str, ids: list):
        super().__init__(name, label, ids, telescope=Telescope.LOW)
        self.n_stations = len(ids)

    def __repr__(self):
        return (
            f"<LowSubarray(name='{self.name}', label='{self.label}',"
            f" n_stations={self.n_stations})>"
        )

    def __eq__(self, other):
        """Equality method. The configuration and attribute is ignored here because
        the same file could have been loaded via a different path. The checksum
        is the real test.
        """
        if not isinstance(other, LowSubarray):
            return False

        return (
            self.name,
            self.ids,
            self.n_stations,
        ) == (
            other.name,
            other.ids,
            other.n_stations,
        )


class MidSubarraySchema(Schema):
    """
    Schema to de/serialize the data of the Subarray class
    """

    name = fields.Str()
    configuration = fields.Str()
    ids = fields.List(fields.Int())
    label = fields.Str()
    md5_checksum = fields.Str()

    @post_load
    def make_subarray(self, data, **kwargs):
        return MidSubarray(**data)


class LowSubarraySchema(Schema):
    """
    Schema to de/serialize the data of the Subarray class
    """

    name = fields.Str()
    ids = fields.List(fields.Int())
    label = fields.Str()

    @post_load
    def make_subarray(self, data, **kwargs):
        return LowSubarray(**data)


class SubarrayStorage:
    """
    Class to handle the storage of subarrays in JSON files
    """

    def __init__(self, telescope: Telescope, storage_path: Path | None = None):
        """
        Initialize the storage area and load files

        :param telescope: SKA Telescope, either MID or LOW.
        :param storage_path: path of the storage area
        """
        # There are only two telescopes, so use a simple if statement rather
        # than overcomplicating it with factories
        if telescope == Telescope.LOW:
            self._deserialiser = LowSubarraySchema()
            self._subarray_cls = LowSubarray
        else:
            self._deserialiser = MidSubarraySchema()
            self._subarray_cls = MidSubarray

        if storage_path is None:
            storage_path = SUBARRAY_STORAGE_PATH / telescope.value

        # create a map of JSON filenames to the content of those files
        # data = {f.stem: json.load(open(f)) for f in storage_path.glob("*.json")}
        data = {}
        for f in storage_path.glob("*.json"):
            with open(f) as jsn:
                data[f.stem] = json.load(jsn)

        # if configuration file is only a file name, assume subarray storage
        # path as base directory
        for jsn in data.values():
            if "configuration" not in jsn:
                continue
            jsn_config = jsn.get("configuration")
            if jsn_config == Path(jsn_config).name:
                jsn["configuration"] = str(storage_path / jsn_config)

        self._data = data

    def list(self) -> list[U]:
        """
        List the Subarray objects for the subarray files stored
        """
        objects = [self._subarray_cls(**v) for v in self._data.values()]
        return sorted(objects, key=operator.attrgetter("label"))

    def load_by_label(self, label) -> U:
        """
        Load one of the subarray files stored

        :param label: label of the subarray configuration
        :type label: str
        """
        subarrays = [
            self._deserialiser.load(config)
            for config in self._data.values()
            if config["label"] == label
        ]

        if len(subarrays) == 0:
            raise ValueError(f"Subarray with label {label} not found")
        if len(subarrays) > 1:
            raise ValueError(f"Multiple subarrays with label {label} found.")

        return subarrays[0]

    def load_by_name(self, name) -> U:
        """
        Load one of the subarray files stored

        :param name: name of the subarray configuration
        :type name: str
        """
        subarrays = [
            self._deserialiser.load(config)
            for config in self._data.values()
            if config["name"] == name
        ]

        if len(subarrays) == 0:
            raise ValueError(f"Subarray with name {name} not found")
        if len(subarrays) > 1:
            raise ValueError(f"Multiple subarrays with name {name} found.")

        return subarrays[0]

    def filename_label_mapping(self):
        # default order for sorting a dict is to sort by dict key
        return {k: v["label"] for k, v in sorted(self._data.items())}
