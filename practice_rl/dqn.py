import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym
import cv2
import ale_py

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # CNN 레이어들
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # CNN 출력 크기 계산
        conv_out_size = self._get_conv_out_size(input_shape)

        # Fully connected 레이어들
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _get_conv_out_size(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class AtariPreprocessing:
    def __init__(self, env_name, frame_skip=4, frame_stack=4, render_mode=None):
        # render_mode 매개변수 추가
        self.env = gym.make(env_name, render_mode=render_mode)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def reset(self):
        state, _ = self.env.reset()
        processed_state = self._preprocess_frame(state)
        for _ in range(self.frame_stack):
            self.frames.append(processed_state)
        return self._get_state()

    def step(self, action):
        total_reward = 0
        for _ in range(self.frame_skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        processed_state = self._preprocess_frame(state)
        self.frames.append(processed_state)
        done = terminated or truncated

        return self._get_state(), total_reward, done, info

    def _preprocess_frame(self, frame):
        # 그레이스케일 변환 및 크기 조정
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized.astype(np.float32) / 255.0

    def _get_state(self):
        return np.array(self.frames, dtype=np.float32)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class DQNAgent:
    def __init__(self, state_shape, n_actions, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_capacity=100000, batch_size=32, target_update=1000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # 신경망 초기화
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 타겟 네트워크 초기화
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 경험 재생 버퍼
        self.memory = ReplayBuffer(buffer_capacity)
        self.step_count = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.max(1)[1].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 다음 Q값 (타겟 네트워크 사용)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 손실 계산 및 역전파
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()

        # 엡실론 감소
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # 타겟 네트워크 업데이트
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']


def train_dqn():
    # 환경 설정
    env_name = "ALE/Breakout-v5"
    env = AtariPreprocessing(env_name, render_mode=None)

    # 에이전트 초기화
    state_shape = (4, 84, 84)  # frame_stack, height, width
    n_actions = env.env.action_space.n
    agent = DQNAgent(state_shape, n_actions)

    # 학습 파라미터
    n_episodes = 100000
    max_steps_per_episode = 100000
    train_frequency = 4
    save_frequency = 100

    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_score = 0
        step = 0

        while step < max_steps_per_episode:
            # 액션 선택
            action = agent.select_action(state)

            # 환경에서 한 스텝 실행
            next_state, reward, done, _ = env.step(action)

            # 경험 저장
            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            episode_score += reward
            step += 1

            # 학습
            if step % train_frequency == 0:
                agent.train()

            if done:
                break

        scores.append(episode_score)

        # 진행상황 출력
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

        # 모델 저장
        if episode % save_frequency == 0 and episode > 0:
            agent.save_model(f"./agent/atari/dqn_model_episode_{episode}.pth")

    env.close()
    return agent, scores


def simulate_trained_agent(model_path, n_episodes=5, render=True):
    """학습된 모델로 게임 시뮬레이션"""

    # 환경 설정 - render_mode 명시
    env_name = "ALE/Breakout-v5"
    render_mode = "human" if render else None
    env = AtariPreprocessing(env_name, render_mode=render_mode)

    # 에이전트 로드
    state_shape = (4, 84, 84)
    n_actions = env.env.action_space.n
    agent = DQNAgent(state_shape, n_actions)
    agent.load_model(model_path)

    test_scores = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_score = 0
        step = 0

        print(f"시뮬레이션 에피소드 {episode + 1} 시작...")

        while True:
            # render() 호출 조건 수정
            if render and hasattr(env.env, 'render'):
                env.render()

            # 학습된 정책으로 액션 선택 (탐험 없음)
            action = agent.select_action(state, training=False)

            # 환경에서 한 스텝 실행
            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_score += reward
            step += 1

            if done:
                break

        test_scores.append(episode_score)
        print(f"에피소드 {episode + 1} 완료 - 점수: {episode_score}")

    env.close()

    print(f"\n시뮬레이션 결과:")
    print(f"평균 점수: {np.mean(test_scores):.2f}")
    print(f"최고 점수: {max(test_scores)}")
    print(f"최저 점수: {min(test_scores)}")

    return test_scores


# 시뮬레이션 실행 (학습된 모델이 있는 경우)
# test_scores = simulate_trained_agent("dqn_model_episode_1000.pth")

def evaluate_agent_performance(agent, env, n_eval_episodes=10):
    """에이전트 성능 평가"""

    eval_scores = []

    for episode in range(n_eval_episodes):
        state = env.reset()
        episode_score = 0

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_score += reward

            if done:
                break

        eval_scores.append(episode_score)

    return {
        'mean_score': np.mean(eval_scores),
        'std_score': np.std(eval_scores),
        'max_score': max(eval_scores),
        'min_score': min(eval_scores),
        'scores': eval_scores
    }


def plot_training_results(scores, window_size=100):
    """학습 결과 시각화"""
    import matplotlib.pyplot as plt

    # 이동 평균 계산
    moving_avg = []
    for i in range(len(scores)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(scores[start_idx:i + 1]))

    plt.figure(figsize=(15, 5))

    # 원본 점수
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Episode Score')
    plt.plot(moving_avg, label=f'{window_size}-Episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()

    # 최근 에피소드들의 분포
    plt.subplot(1, 2, 2)
    recent_scores = scores[-200:] if len(scores) > 200 else scores
    plt.hist(recent_scores, bins=20, alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution (Recent Episodes)')

    plt.tight_layout()
    plt.show()


# 학습 실행
# TODO: 바이크코딩 딸깍 ㅋㅋㅋㅋ, Is it right the factor I can improve is only the hyperparameters such as episode?
if __name__ == "__main__":
    # agent, scores = train_dqn()
    #
    # # 점수 그래프 그리기
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(12, 6))
    # plt.plot(scores)
    # plt.title('DQN Training Scores')
    # plt.xlabel('Episode')
    # plt.ylabel('Score')
    # plt.show()
    test_scores = simulate_trained_agent("./agent/atari/dqn_model_episode_11000.pth")
