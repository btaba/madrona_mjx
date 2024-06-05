import jax
import jax.numpy as jp
import mujoco

from typing import Optional, Any, List, Sequence, Dict, Tuple, Union, Callable

from brax import base
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import mjcf

from etils import epath

from madrona_mjx.renderer import BatchRenderer


def _load_sys(path: str) -> base.System:
  path = epath.Path(path)
  xml = path.read_text()
  assets = {}
  for f in path.parent.glob('*.xml'):
    assets[f.name] = f.read_bytes()
    for f in (path.parent / 'assets').glob('*'):
      assets[f.name] = f.read_bytes()
  for f in epath.Path('/usr/local/google/home/btaba/.maniskill/data/robots/anymal_c/meshes').glob('*'):
    assets[f.name] = f.read_bytes()
  model = mujoco.MjModel.from_xml_string(xml, assets)
  return mjcf.load_model(model)


class AnymalEnv(PipelineEnv):
  """Environment for training aloha to bring an object to target."""

  def __init__(self, render_batch_size: int, gpu_id: int = 0,
               width: int = 128, height: int = 128,
               add_cam_debug_geo: bool = False, 
               render_viz_gpu_hdls = None, **kwargs):
    sys = _load_sys('/usr/local/google/home/btaba/.maniskill/data/robots/anymal_c/urdf/anymal.xml')
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)
    # Madrona renderer
    print('Triangles: ', sys.mj_model.nmeshface)
    print('Nmesh: ', sys.mj_model.nmesh)
    self.renderer = BatchRenderer(sys, gpu_id, render_batch_size, 
                                  width, height, add_cam_debug_geo,
                                  render_viz_gpu_hdls)

  def reset(self, rng: jax.Array) -> State:
    rng, rng_target, rng_box = jax.random.split(rng, 3)
    pipeline_state = self.pipeline_init(jp.array(self.sys.mj_model.qpos0), jp.zeros(self.sys.nv))
    info = {'rng': rng}
    render_token, rgb, depth = self.renderer.init(pipeline_state)
    #print(rgb.shape, depth.shape)
    info.update({'render_token': render_token, 'rgb': rgb[0], 'depth': depth[0]})
    obs = jp.ones(5)
    reward = 1.0
    done = 0.0
    metrics = {}
    state = State(pipeline_state, obs, reward, done, metrics, info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    del action
    # data = self.pipeline_step(state.pipeline_state, action)
    data = state.pipeline_state
    render_token, rgb, depth = self.renderer.render(state.info['render_token'], data)
    state.info.update({'render_token': render_token, 'rgb': rgb, 'depth': depth})
    state = State(data, state.obs, state.reward, state.done, state.metrics, state.info)
    return state
