import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from coqpit import Coqpit


class LanguageManager:
    """Manage the languages for multi-language 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by language or clip.

    For now there is on scenario considered:

    1. Models using language embedding layers. The datafile only maps language names to ids used by the embedding layer.

    Args:
        language_id_file_path (str, optional): Path to the metafile that maps language names to ids used by
        TTS models. Defaults to "".
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        language_id_file_path: str = "",
    ):

        self.data_items = []
        self.d_vectors = {}
        self.language_ids = {}
        self.clip_ids = []

        if data_items:
            self.language_ids, self.language_names, _ = self.parse_languages_from_data(self.data_items)

        if language_id_file_path:
            self.set_language_ids_from_file(language_id_file_path)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with open(json_file_path) as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_languages(self):
        return len(self.language_ids)

    @property
    def language_names(self):
        return list(self.language_ids.keys())

    @property
    def d_vector_dim(self):
        """Dimensionality of d_vectors. If d_vectors are not loaded, returns zero."""
        if self.d_vectors:
            return len(self.d_vectors[list(self.d_vectors.keys())[0]]["embedding"])
        return 0

    @staticmethod
    def parse_languages_from_data(items: list) -> Tuple[Dict, int]:
        """Parse language IDs from data samples retured by `load_meta_data()`.

        Args:
            items (list): Data sampled returned by `load_meta_data()`.

        Returns:
            Tuple[Dict, int]: language IDs and number of languages.
        """
        languages = sorted({item[2] for item in items})
        language_ids = {name: i for i, name in enumerate(languages)}
        num_languages = len(language_ids)
        return language_ids, num_languages

    def set_language_ids_from_data(self, items: List) -> None:
        """Set language IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_meta_data()`.
        """
        self.language_ids, _ = self.parse_languages_from_data(items)

    def set_language_ids_from_file(self, file_path: str) -> None:
        """Set language IDs from a file.

        Args:
            file_path (str): Path to the file.
        """
        self.language_ids = self._load_json(file_path)

    def save_language_ids_to_file(self, file_path: str) -> None:
        """Save language IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.language_ids)

    def get_languages(self) -> List:
        return self.language_ids


def _set_file_path(path):
    """Find the languages.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "languages.json")
    path_continue = os.path.join(path, "languages.json")
    if os.path.exists(path_restore):
        return path_restore
    if os.path.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `languages.json` not found in {path}")


def load_language_mapping(out_path):
    """Loads language mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with open(json_file) as f:
        return json.load(f)


def save_language_mapping(out_path, language_mapping):
    """Saves language mapping if not yet present."""
    if out_path is not None:
        languages_json_path = _set_file_path(out_path)
        with open(languages_json_path, "w") as f:
            json.dump(language_mapping, f, indent=4)


def get_language_manager(
    c: Coqpit, data: List = None, restore_path: str = None, out_path: str = None
) -> LanguageManager:
    """Initiate a `LanguageManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        data (List): Data samples used in training to infer languages from. It must be provided if language embedding
            layers is used. Defaults to None.
        out_path (str, optional): Save the generated language IDs to a output path. Defaults to None.

    Returns:
        LanguageManager: initialized and ready to use instance.
    """
    language_manager = LanguageManager()
    if c.use_language_embedding:
        if data is not None:
            language_manager.set_language_ids_from_data(data)
        if restore_path:
            languages_file = _set_file_path(restore_path)
            # restoring language manager from a previous run.
            language_ids_from_data = language_manager.language_ids
            language_manager.set_language_ids_from_file(languages_file)
            assert all(
                language in language_manager.language_ids for language in language_ids_from_data
            ), " [!] You cannot introduce new languages to a pre-trained model."
        print(
            " > Training with {} languages: {}".format(
                language_manager.num_languages, ", ".join(language_manager.language_ids)
            )
        )
        # save file if path is defined
        if out_path:
            out_file_path = os.path.join(out_path, "languages.json")
            print(f" > Saving `languages.json` to {out_file_path}.")
            language_manager.save_language_ids_to_file(out_file_path)
    return language_manager
