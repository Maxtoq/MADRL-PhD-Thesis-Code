from multiagent.environment import MultiAgentEnv
import numpy as np


class RenderMultiAgent(MultiAgentEnv):
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, discrete_action=False, 
                 shared_viewer=True):
        super().__init__(world, reset_callback, reward_callback,
                 observation_callback, info_callback,
                 done_callback, discrete_action, 
                 shared_viewer)

    # render environment
    def render(self, range1=False, range2=False, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            agent = 0
            for entity in self.world.entities:
                
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom = rendering.make_circle(entity.size)
                    geom.set_color(*entity.color, alpha=0.5)
                    if (agent == 0 and range1 == True) or (agent == 1 and range2 == True):
                        vision = rendering.make_circle(0.4)
                        vision.set_color(*entity.color, alpha=0.2)
                        vision.add_attr(xform)
                        self.render_geoms.append(vision)
                    agent += 1
                else:
                    if entity.shape == "circle":
                        geom = rendering.make_circle(entity.size)
                    elif entity.shape == "square":
                        geom = rendering.make_square(entity.size, entity.size)
                    elif entity.shape == "triangle":
                        geom = rendering.make_triangle(entity.size, entity.size)

                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        self._reset_render()

        return results
