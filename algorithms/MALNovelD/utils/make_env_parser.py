import numpy as np
import json
import imp
import os
import re
from pathlib import Path
from shutil import copyfile

def make_env(args, sce_conf={}, discrete_action=False):
    from utils.render_multiagent import RenderMultiAgent

    # load scenario from script
    scenar_lib = imp.load_source('', args.env_path)
    scenario = scenar_lib.Scenario()

    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = RenderMultiAgent(world, scenario.reset_world, scenario.reward,
                        scenario.observation, 
                        done_callback=scenario.done if hasattr(scenario, "done")
                        else None, discrete_action=discrete_action)

    # If world has an attribut objects
    colors = []
    shapes = []
    if hasattr(env.world, 'objects'):
        # Get the color and the shape
        for object in env.world.objects :
                colors.append(object.num_color)
                shapes.append(object.num_shape)
    else:
        print('No objects')

    # Get parser
    if args.parser == "basic":
        parser = scenar_lib.ObservationParser(args, colors, shapes)
    if args.parser == 'strat':
        parser = scenar_lib.ObservationParserStrat(args, sce_conf, colors, shapes)

    return env, parser

def load_scenario_config(config, run_dir):
    sce_conf = {}
    if config.sce_conf_path is not None:
        copyfile(config.sce_conf_path, run_dir / 'sce_config.json')
        with open(config.sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf, '\n')
    return sce_conf