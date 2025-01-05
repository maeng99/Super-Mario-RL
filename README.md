<img src="https://github.com/user-attachments/assets/830d617a-3f18-4c90-a49e-5e31931b38c3" width="100%"/>

# 🍄강화학습으로 슈퍼마리오 게임하기🍄
<img src="https://github.com/user-attachments/assets/d1ab96f1-cd6e-402c-b21a-52c84297ecad" width="300px"/>
<br/>

## 1. Project Overview
- **프로젝트 이름: Super Mario Bros Reinforcement Learning**
- 프로젝트 설명: Open AI Gym 환경을 기반으로 DDQN(Double Deep Q-Network)을 사용하여 슈퍼마리오 게임을 수행
<br/>

## 2. Team Members
| 맹의현 | 진영인 |
|:------:|:------:|
| [GitHub](https://github.com/maeng99) | [GitHub](https://github.com/) |
<br/>

## 3. Environment
### 3.1 gym-super-mario-bros
- Super Mario Bros를 위한 Open AI Gym Open AI Gym 환경
```python
%pip install gym-super-mario-bros==7.4.0
```
<img src="https://github.com/user-attachments/assets/577cb683-7120-469c-9fc7-b02718ebe5fb" width="400px" />

### 3.2 Action space
- 보유한 자원 내에서 효율적인 학습을 위해 action space를 제한
    - { 0: Right, 1: Right+A(jump) }
- SkipFrame: 일정 frame동안 동일 action 반복하여 reward를 누적한 후 한번에 반환
```python
env = JoypadSpace(env, [
    ["right"], ["right", "A"]
])
```
```python
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """모든 `skip` 프레임만 반환합니다."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """행동을 반복하고 포상을 더합니다."""
        total_reward = 0.0
        for i in range(self._skip):
            # 포상을 누적하고 동일한 작업을 반복합니다.
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info
```
### 3.3 Observation space:
    - GrayScale: observation space을 흑백 변환
    - Resize: observation space를 고정된 크기(84 X 84)로 변환
```python
class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # [H, W, C] 배열을 [C, H, W] 텐서로 바꿉니다.
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
```
<br />

## 4. Agent
<img src="https://github.com/user-attachments/assets/fc79c23d-f8d5-4f81-bd67-d6b58754ff64" width="400px" />

### 4.1 Act: 현재 state를 기반으로 action policy에 따라 선택
- epsilon-greedy action 선택
```python
class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 마리오의 DNN은 최적의 행동을 예측합니다 - 이는 학습하기 섹션에서 구현합니다.
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # Mario Net 저장 사이의 경험 횟수

    def act(self, state):
        """
    주어진 상태에서, 입실론-그리디 행동(epsilon-greedy action)을 선택하고, 스텝의 값을 업데이트 합니다.

    입력값:
    state (``LazyFrame``): 현재 상태에서의 단일 상태(observation)값을 말합니다. 차원은 (state_dim)입니다.
    출력값:
    ``action_idx`` (int): Mario가 수행할 행동을 나타내는 정수 값입니다.
    """
        # 임의의 행동을 선택하기
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 최적의 행동을 이용하기
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # exploration_rate 감소하기
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # 스텝 수 증가하기
        self.curr_step += 1
        return action_idx
```
### 4.2 Remember
- cache를 통해 메모리에 경험을 추가하고, recall을 통해 경험을 메모리에서 불러와 사용
```python
class Mario(Mario):  # 연속성을 위한 하위 클래스입니다.
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 64

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        입력값:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        메모리에서 일련의 경험들을 검색합니다.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
```
### 4.3 Learn
- 최적의 action을 위한 Q-function 업데이트
- chapter 5에서 계속..
<br />

## 5. Learn
### 5.1 MarioNet(DDQN)
- **DDQN**: action의 선택(online 네트워크)과 평가(target 네트워크)를 분리
    - Q-value의 신뢰성 향상 / overestimate 문제 완화
```python
class MarioNet(nn.Module):
    """작은 CNN 구조
  입력 -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> 출력
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target 매개변수 값은 고정시킵니다.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
```
### 5.2 TD_estimate & TD_target
- **TD_estimate**: 현재 state에서 선택된 action에 해당하는 Q값 추출
- **TD_target**: online 네트워크로 다음 state에서 최대 Q값을 가지는 action 선택해,<br/>target 네트워크로 최대 Q값을 평가
```python
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.gamma = 0.9

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
```
### 5.3 Update Model
- **update_Q_online()** : td_estimate와 td_target 간의 loss를 계산해 가중치를 업데이트
- **sync_Q_target()** : target 네트워크의 가중치를 online 네트워크와 동기화
```python
class Mario(Mario):
    def __init__(self, state_dim, action_dim, save_dir):
        super().__init__(state_dim, action_dim, save_dir)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        # Learning rate scheduler 추가
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.9  # 10,000 스텝마다 학습률 90%로 감소
        )
        self.loss_fn = torch.nn.SmoothL1Loss()


    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
```
<br />

## 6. Train
- 게임 종료 시까지 { state -> action -> cache -> learn -> log -> next_state } 반복
```python
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

#save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / "epi10000lrchaction4_2"
save_dir.mkdir(parents=True)

mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

logger = MetricLogger(save_dir)

episodes = 000
for e in range(episodes):

    state = env.reset()

    # 게임을 실행시켜봅시다!
    while True:

        # 현재 상태에서 에이전트 실행하기
        action = mario.act(state)

        # 에이전트가 액션 수행하기
        next_state, reward, done, trunc, info = env.step(action)

        # 기억하기
        mario.cache(state, next_state, action, reward, done)

        # 배우기
        q, loss = mario.learn()

        # 기록하기
        logger.log_step(reward, loss, q)

        # 상태 업데이트하기
        state = next_state

        # 게임이 끝났는지 확인하기
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if (e % 20 == 0) or (e == episodes - 1):
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
```
<br/>

## 7. Experiment
### 7.1 Using Scheduler
- ReduceLROnPlateau 스케줄러 / StopRL 스케줄러 적용
```python
# Learning rate scheduler 추가
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.9  # 10,000 스텝마다 학습률 90%로 감소
        )
```
- 보다 안정적이긴하지만, 큰 차이 없음
### 7.2 Action Space Extention
- 좀 더 다양한 action을 사용하도록 확장
<img src="https://github.com/user-attachments/assets/010fa0c9-0788-47c4-a613-cd625ffcd4ca" width="400px" />

```python
env = JoypadSpace(env, [
    ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"]
])
```
- 보다 안정적이긴하지만, 큰 차이 없음
<br/>

## 8. Discussion
- 보유 자원의 부족으로 인한 학습의 한계가 있어 너무 아쉬웠음
<br />

## 9. Presentation PDF
- 프로젝트 자세히 알아보기

[📄 SuperMarioBros_RL_PDF](https://github.com/maeng99/SuperMarioBros_RL/blob/main/SuperMarioBros_RL_pdf.pdf)


