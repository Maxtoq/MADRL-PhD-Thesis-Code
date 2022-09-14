import json
import imp
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

    # Create parser
    parser_args = [
        sce_conf['nb_agents'], 
        sce_conf['nb_objects'], 
        args.chance_not_sent]
    parser = scenar_lib.ObservationParser(*parser_args)

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