'''
Author: Radillus
Date: 2023-05-23 16:07:10
LastEditors: Radillus
LastEditTime: 2023-07-20 22:37:29
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import time
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from gymnasium import spaces, Env
from matplotlib.animation import FuncAnimation


class TopoDesignEnv(Env):
    metadata = {
        'render_modes': ['matplotlib_realtime', 'matplotlib_record'],
    }
    
    def __init__(self, env_config) -> None:
        self.surface_shape:tuple = env_config['suface_shape']
        self.wavefront_shape:tuple = env_config['wavefront_shape']
        self.render_mode:str = env_config.get('render_mode', None)
        self.continual_steps:int = env_config.get('continual_steps', 1)
        self.get_target_func = env_config['get_target_func']
        self.get_result_func = env_config['get_result_func']
        self.stop_threshold:float = env_config.get('stop_threshold', 1e-3)
        self.max_steps:int = env_config.get('max_steps', None)
        self.debug:bool = env_config.get('debug', False)
        if self.render_mode == 'matplotlib_record':
            self.render_flag = False
            self.record_times:int = env_config.get('record_times', 100)
            self.pl_file_path = None
            if self.max_steps is not None:
                self.record_times = min(self.record_times, self.max_steps)
            self.record_times = max(self.record_times, 500)  # max record times for matplotlib
        
        if isinstance(self.surface_shape, (int, np.signedinteger, np.unsignedinteger)):
            self.surface_shape = (self.surface_shape, self.surface_shape)
        if isinstance(self.wavefront_shape, (int, np.signedinteger, np.unsignedinteger)):
            self.wavefront_shape = (self.wavefront_shape, self.wavefront_shape)
        assert len(self.surface_shape) == 2 and self.surface_shape[0] == self.surface_shape[1]
        assert len(self.wavefront_shape) == 2 and self.wavefront_shape[0] == self.wavefront_shape[1]
        
        self.big_size = max(self.surface_shape[0], self.wavefront_shape[0])
        self.observation_size = (self.big_size, self.big_size, 3)
        
        self.observation_space = spaces.Box(low=-10, high=10, shape=self.observation_size)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5*self.continual_steps,))
        
        self.step_count = 0
        
        if self.render_mode == 'matplotlib':
            plt.ion()
            self.fig = plt.figure()
            self.surface_ax = self.fig.add_subplot(2, 2, 1)
            self.loss_ax = self.fig.add_subplot(2, 2, 2)
            self.wavefront_ax = self.fig.add_subplot(2, 2, 3)
            self.target_wavefront_ax = self.fig.add_subplot(2, 2, 4)
            self.target_wavefront_ax.grid()
            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.01)
    
    def _get_obs(self):
        # print('sf shape', self.surface.shape)
        # print('wf shape', self.wavefront.shape)
        if self.surface_shape[0] < self.big_size:
            scale_surface = zoom(self.surface, self.big_size / self.surface_shape[0])
            scale_wavefront = self.wavefront
            scale_target_wavefront = self.target_wavefront
        elif self.wavefront_shape[0] > self.big_size:
            scale_surface = self.surface
            scale_wavefront = zoom(self.wavefront, self.big_size / self.wavefront_shape[0])
            scale_target_wavefront = zoom(self.target_wavefront, self.big_size / self.wavefront_shape[0])
        return np.stack([scale_surface, scale_wavefront, scale_target_wavefront], axis=2).astype(np.float32)
        
    def _get_info(self):
        return {
            'wavefront_loss': self.wavefront_loss[-1],
        }
    
    def _clear_render(self):
        self.wavefront_ax.clear()
        self.target_wavefront_ax.clear()
        self.surface_ax.clear()
        self.loss_ax.clear()
    
    def _render_frame(self):
        self.surface_ax.imshow(self.surface)
        self.wavefront_ax.imshow(self.wavefront)
        self.target_wavefront_ax.imshow(self.target_wavefront)
        self.loss_ax.plot(self.wavefront_loss)
        self.fig.canvas.draw()
        plt.pause(0.01)
        # plt.draw()
    
    def _get_loss(self):
        return np.mean(np.abs(self.wavefront - self.target_wavefront))
    
    def reset(self, *, seed=None, options=None):
        if self.render_mode == "matplotlib_record" and self.render_flag and self.pl_file_path is not None and self.now_step != 0:
            self.generate_animation()
        
        target_wavefront_ref = self.get_target_func()
        self.target_wavefront = target_wavefront_ref
        
        self.surface = np.random.randn(*self.surface_shape)
        self.wavefront = None
        self._update_wavefront()
        
        self.wavefront_loss = [self._get_loss()]
        self.step_count = 0
        if self.render_mode == 'matplotlib_realtime':
            self._clear_render()
            self._render_frame()
        elif self.render_mode == "matplotlib_record":
            now_time = time.localtime()
            now_minute = now_time.tm_min
            self.pl_file_path = './data/images/'+time.strftime('%m-%d-%H-',now_time)+str(now_minute//1)
            if os.path.exists(self.pl_file_path) and not (self.render_flag and self.now_step == 0):
                self.render_flag = False
            else:
                self.render_flag = True
                self.actual_record_times = 0
                with open(self.pl_file_path,'w') as f:
                    f.write('This is a placeholder file')
                f.close()
                self.start_time_str = time.strftime('%m-%d-%H-%M', time.localtime())
                self.record_interval = self.max_steps // self.record_times
                self.surface_record = np.zeros((self.record_times+1, *self.surface_shape))
                self.wavefront_record = np.zeros((self.record_times+1, *self.wavefront_shape))
                self.target_wavefront_record = np.zeros((self.record_times+1, *self.wavefront_shape))
                self._record()
        return self._get_obs(), self._get_info()
    
    def _record(self):
        record_id = self.step_count // self.record_interval
        self.surface_record[record_id,...] = self.surface
        self.wavefront_record[record_id,...] = self.wavefront
        self.target_wavefront_record[record_id,...] = self.target_wavefront
    
    def _convert_action_to_index(self, value, high, low=0):
        # contianed in [low, high)
        assert value >= -1 and value <= 1
        return int((value + 1) / 2 * (high - low))
    
    def _caculate_start_end(self, central, size, ax_size):
        central = self._convert_action_to_index(central, ax_size)
        size = self._convert_action_to_index(size, ax_size)
        start = max(0, central - size // 2)
        end = min(ax_size, central + size // 2 + 1)
        return int(start), int(end)
    
    def _add_rectangle(self, rectangle_attribute):
        rectangle_x_central, rectangle_x_size, rectangle_y_central, rectangle_y_size, rectangle_value = rectangle_attribute
        rectangle_x_start, rectangle_x_end = self._caculate_start_end(rectangle_x_central, rectangle_x_size, self.surface_shape[0])
        rectangle_y_start, rectangle_y_end = self._caculate_start_end(rectangle_y_central, rectangle_y_size, self.surface_shape[1])
        self.surface[rectangle_x_start:rectangle_x_end, rectangle_y_start:rectangle_y_end] += rectangle_value
        self.surface = np.clip(self.surface, -1, 1)
        
    def _update_wavefront(self) -> float:
        wavefront_ref = self.get_result_func(self.surface)
        now_wavefront = wavefront_ref
        if self.wavefront is None:
            self.wavefront = now_wavefront
            return None
        else:
            reward = np.mean(np.abs(self.target_wavefront - self.wavefront)) - np.mean(np.abs(self.target_wavefront - now_wavefront))
            self.wavefront = now_wavefront
            return reward

    def _is_terminated(self):
        return np.max(np.abs(self.wavefront - self.target_wavefront)) < self.stop_threshold

    def _is_truncated(self):
        if self.max_steps is None:
            return False
        else:
            return self.step_count > self.max_steps
    
    def _generate_animation(self):
        print(f'---------------animation generate at {self.pl_file_path}.mp4-----------------')
        self.render_flag = False
        end_idx = self.step_count // self.record_interval
        fig = plt.figure()
        surface_ax = fig.add_subplot(2,2,1)
        loss_ax = fig.add_subplot(2,2,2)
        wavefront_ax = fig.add_subplot(2,2,3)
        target_wavefront_ax = fig.add_subplot(2,2,4)
        surface_ax.set_title('surface')
        loss_ax.set_title('loss')
        wavefront_ax.set_title('wavefront')
        target_wavefront_ax.set_title('target wavefront')
        
        def update(frame):
            surface_ax.clear()
            loss_ax.clear()
            wavefront_ax.clear()
            target_wavefront_ax.clear()
            surface_ax.imshow(self.surface_record[frame,...])
            loss_ax.plot(self.wavefront_loss[:frame])
            wavefront_ax.imshow(self.wavefront_record[frame,...])
            target_wavefront_ax.imshow(self.target_wavefront_record[frame,...])
            fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=end_idx, interval=200, cache_frame_data=False)
        ani.save(self.pl_file_path+'.mp4', writer='ffmpeg')
        plt.close('all')
    
    def step(self, action:np.ndarray):
        assert self.action_space.contains(action)
        action = action.reshape((-1,5))
        for rectangle_attribute in action:
            self._add_rectangle(rectangle_attribute)
        reward = self._update_wavefront()
        self.wavefront_loss.append(self._get_loss())
        self.step_count += 1
        if self.render_mode == 'matplotlib_realtime':
            self._clear_render()
            self._render_frame()
        elif self.render_mode == 'matplotlib_record' and self.render_flag:
            self._record()
        if self.debug:
            print('***************************************************************')
            print('step count', self.step_count)
            print('wavefront loss', self.wavefront_loss[-1])
            print('reward', reward)
            print('***************************************************************')
        return self._get_obs(), reward, self._is_truncated(), self._is_terminated(), self._get_info()
    
    def close(self):
        self._generate_animation()

if __name__ == '__main__':

    
    sf_size = 100
    wf_size = 3 * sf_size
    
    def get_target():
        a = np.zeros((wf_size, wf_size))
        a[0:wf_size//3, 0:wf_size//3] = 0.5
        return a

    def get_result(a:np.ndarray):
        a = zoom(a, 3)
        # t = np.fft.fft2(a)
        # t = np.fft.fftshift(t)
        # t = t @ t
        # t = np.fft.ifftshift(t)
        # t = np.fft.ifft2(t)
        # t = np.abs(t) / (np.max(np.abs(t))+1e-6)
        return a


    env_config = {
        'suface_shape': sf_size,
        'wavefront_shape': wf_size,
        'render_mode': 'matplotlib',
        'continual_steps': 1,
        'get_target_func': get_target,
        'get_result_func': get_result,
        'stop_threshold': 1e-3,
        'max_steps': 10000,
    }
        
    env_instance = TopoDesignEnv(env_config=env_config)
    env_instance.reset()
    
    # observation_space = env_instance.observation_space
    # reset_obs = env_instance.reset()[0]
    
    # print(np.can_cast(reset_obs.dtype, observation_space.dtype))
    # print(reset_obs.shape == observation_space.shape)
    # print(np.all(reset_obs >= observation_space.low))
    # print(np.all(reset_obs <= observation_space.high))
    
    # print('obs shape: ', observation_space.sample().shape)
    # print('obs type: ', type(env_instance.observation_space.sample()))
    # print('reset shape: ', reset_obs.shape)
    # print('reset type: ', type(env_instance.reset()[0]))
    # print('is contained: ', observation_space.contains(reset_obs))
    for i in range(100):
        print(i)
        env_instance.step(env_instance.action_space.sample())
