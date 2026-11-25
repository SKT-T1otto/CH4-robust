"""Stage-5 collection contract, registered scene identities and private ordering."""
import copy
from dataclasses import asdict, dataclass

import numpy as np

from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.config import scenario_fingerprint
from .config import DEFAULT_CONFIG, ROOT, file_sha, load, read_json
from .scenarios import validate_registry


TRAIN_PROTOCOL = ROOT/'configs/ch4/nominal_maddpg_training.json'
STREAMS = ('initialization', 'scene_order', 'exploration', 'replay')


def stream_seeds(seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('run seed must be a nonnegative uint32 integer')
    values = np.random.SeedSequence([seed, 51025]).generate_state(len(STREAMS))
    return dict(zip(STREAMS, (int(v) for v in values)))


def load_training_protocol():
    p, c = read_json(TRAIN_PROTOCOL), load()
    if (p['schema'] != 'auv.ch4.nominal_maddpg.collection.v1' or p['stage'] != 5
            or p['task_training_enabled'] is not False or p['maximum_prefix_episodes'] != 8
            or p['maximum_steps_per_prefix'] != 32 or p['maximum_updates_per_model'] != 4
            or p['training_checkpoints_written'] != 0
            or p['allowed_evaluation_groups'] != ['validation']
            or p['nominal_admission_opened'] is not False or p['mismatch_sets_opened'] is not False):
        raise ValueError('unexpected bounded collection protocol')
    for name, sha in p['input_files'].items():
        if file_sha(ROOT/name) != sha:
            raise ValueError('collection reference changed: '+name)
    expected = dict(type='OU', initial_sigma=c['learning']['exploration']['initial_sigma'],
        minimum_sigma=c['learning']['exploration']['minimum_sigma'],
        decay_per_environment_step=c['learning']['exploration']['decay_per_environment_step'],
        theta=.15, dt=1., mean=0., reset='memory_only_at_episode_start',
        clock='one_tick_per_training_team_decision_including_closed_gates',
        disabled='zero_noise_no_draw_no_memory_change_clock_advances')
    if p['exploration'] != expected:
        raise ValueError('registered OU exploration changed')
    return p


@dataclass(frozen=True)
class SceneOrigin:
    catalog_sha256: str
    scenario_id: str
    scenario_seed: int
    group: str
    scene_sha256: str
    physical_fingerprint: str
    episode_id: int

    def as_dict(self):
        return asdict(self)


class SceneCatalog:
    def __init__(self):
        self.config = load()
        path = ROOT/self.config['scene_manifest']
        self.registry = read_json(path)
        validate_registry(self.config, self.registry, DEFAULT_CONFIG)
        self.sha256 = file_sha(path)
        self.entries = {entry['scene']['scenario_id']: copy.deepcopy(entry)
                        for group in self.registry['groups'].values() for entry in group}

    def scene(self, group, index):
        return copy.deepcopy(self.registry['groups'][group][index]['scene'])

    def origin(self, scene, episode_id, *, allowed_groups):
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError('episode id must be a nonnegative integer')
        known = self.entries.get(scene.get('scenario_id'))
        if known is None or digest(scene) != digest(known['scene']):
            raise ValueError('scene does not match the registered geometry and identity')
        group = scene['ch4_nominal_group']
        if group not in allowed_groups:
            raise ValueError('scene group is not allowed for this data stream: '+group)
        return SceneOrigin(self.sha256, scene['scenario_id'], scene['scenario_seed'], group,
                           digest(scene), scenario_fingerprint(scene), episode_id)

    def validate_origin(self, origin, *, allowed_groups):
        if not isinstance(origin, SceneOrigin):
            raise ValueError('Replay source must be a registered SceneOrigin')
        entry = self.entries.get(origin.scenario_id)
        if entry is None:
            raise ValueError('unknown Replay scene source')
        expected = self.origin(entry['scene'], origin.episode_id, allowed_groups=allowed_groups)
        if origin != expected:
            raise ValueError('Replay source identity differs from the registered scene')


class SceneOrder:
    """Private deterministic permutations, without drawing from the environment RNG."""
    def __init__(self, catalog, seed):
        self.catalog = catalog
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(len(catalog.registry['groups']['train'])).tolist()
        self.cursor, self.epoch = 0, 0

    def next_scene(self):
        if self.cursor == len(self.order):
            self.order = self.generator.permutation(len(self.order)).tolist()
            self.cursor, self.epoch = 0, self.epoch+1
        index = self.order[self.cursor]
        self.cursor += 1
        return self.catalog.scene('train', index)

    def state_dict(self):
        return dict(catalog_sha256=self.catalog.sha256, order=self.order.copy(), cursor=self.cursor,
                    epoch=self.epoch, rng=copy.deepcopy(self.generator.bit_generator.state))
