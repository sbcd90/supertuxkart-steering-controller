import pystk
from supertuxcart_serde import save_dict

def _initialize_pystk():
    cfg = pystk.GraphicsConfig.ld()
    cfg.screen_width = 128
    cfg.screen_height = 96
    try:
        pystk.init(cfg)
    except ValueError as e:
        raise ValueError("Try re-running if using a notebook") from e

class SequentialRollout:
    def __init__(self, callback=None, max_steps=100, frame_skip=1, warmup=20,
                 track_name="lighthouse", step_size=0.1, data_dir="data"):
        self.callback = callback
        self.max_steps = max_steps
        self.frame_skip = frame_skip

        race_config = pystk.RaceConfig(track=track_name, step_size=step_size)
        race_config.num_kart = 1

        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.restart()

        self.track = pystk.Track()
        self.state = pystk.WorldState()
        self.action = pystk.Action()
        self.data_dir = data_dir

        for _ in range(warmup):
            self.race.step()
            self.state.update()
        self.step_idx = 0
        self.max_distance = 0.0

    def _calculate_rewards_and_update_max_dist(self):
        reward = 0
        kart = self.state.karts[0]
        kart_distance_down_track = kart.distance_down_track
        if kart_distance_down_track > self.max_distance:
            reward += (kart_distance_down_track - self.max_distance)
            self.max_distance = kart_distance_down_track
        return reward

    def step(self):
        if self.step_idx >= self.max_steps:
            return None

        self.state.update()
        self.track.update()

        payload = {
            "state": self.state,
            "track": self.track,
            "render_data": self.race.render_data,
        }
        reward = self._calculate_rewards_and_update_max_dist()

        if self.callback is not None:
            state_dict, action_dict = self.callback(**payload)

            state_action_reward_dict = {
                "image": state_dict["image"],
                "velocity": state_dict["velocity"],
                "reward": reward,
                "steer": action_dict["steer"],
                "acceleration": action_dict["acceleration"],
                "brake": action_dict["brake"],
            }
            save_dict(state_action_reward_dict, f"{self.data_dir}/{self.step_idx}.pt")
            for k, v in action_dict.items():
                setattr(self.action, k, v)

        for _ in range(self.frame_skip):
            self.race.step(self.action)
        self.step_idx += 1
        return payload

    def close(self):
        self.race.stop()
        del self.race

if not globals().get("_pystk_init"):
    _initialize_pystk()
    globals()["_pystk_init"] = True