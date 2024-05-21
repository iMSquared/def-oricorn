#!/usr/bin/env python3
"""Pybullet client.

Adapted from:
    bullet3/examples/pybullet/gym/pybullet_utils/bullet_client.py
"""

import functools
import inspect
import pybullet as pb
from typing import Iterable


class BulletClient(object):
  """A wrapper for pybullet to manage different clients."""

  def __init__(self, sim_id: int):
    """Creates a Bullet client from an existing simulation.

    Args:
        sim_id: The client connection id.
    """
    self.__sim_id: int = sim_id

  def __getattr__(self, name: str):
    """Inject the client id into Bullet functions."""
    attr = getattr(pb, name)
    if inspect.isbuiltin(attr):
        pb_attr = attr
        attr = functools.partial(attr,
                                 physicsClientId=self.__sim_id)
        # I think this is fine ...
        functools.update_wrapper(attr, pb_attr)
    return attr

  @property
  def sim_id(self) -> int:
      return self.__sim_id


def main():
    sim_id = pb.connect(pb.DIRECT)
    bc = BulletClient(sim_id)


if __name__ == '__main__':
    main()
