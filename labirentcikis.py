import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Maze tanımı (değişmedi)
MAZE = [
    list("############"),
    list("#          #"),
    list("#  ######  #"),
    list("#     G    #"),
    list("############"),
]

START_POS = (1, 1)
GOAL_POS = (6, 3)

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']

# --- Durum için çevre bilgisini çıkarıyoruz ---
def get_surroundings(pos):
    x, y = pos
    surroundings = []
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            nx, ny = x+dx, y+dy
            if dx == 0 and dy == 0:
                continue
            if 0 <= ny < len(MAZE) and 0 <= nx < len(MAZE[0]):
                surroundings.append(0 if MAZE[ny][nx] == ' ' or MAZE[ny][nx] == 'G' else 1)
            else:
                surroundings.append(1)  # Maze dışı duvar gibi düşün
    return surroundings  # 8 elemanlı liste (0 veya 1)

# --- Ağ ---
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),  # 2 pozisyon + 8 çevre
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

# --- Ortam ---
class MazeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = list(START_POS)
        return self.get_state()

    def get_state(self):
        # x,y pozisyon normalizasyonu + çevre durumu (0-1)
        pos_norm = [self.pos[0]/(len(MAZE[0])-1), self.pos[1]/(len(MAZE)-1)]
        surroundings = get_surroundings(self.pos)
        state = torch.tensor(pos_norm + surroundings, dtype=torch.float32)
        return state

    def step(self, action):
        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1

        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy

        # Default: küçük negatif ceza yok, böylece keşif artar
        reward = 0.0
        done = False

        if MAZE[new_y][new_x] != '#':
            self.pos = [new_x, new_y]
            if (new_x, new_y) == GOAL_POS:
                reward = 10.0
                done = True
        else:
            reward = -1.0  # Duvara çarptı cezası var ama done değil, reset yok

        return self.get_state(), reward, done

    def render(self):
        for y in range(len(MAZE)):
            row = ""
            for x in range(len(MAZE[0])):
                if self.pos == [x,y]:
                    row += "A"
                else:
                    row += MAZE[y][x]
            print(row)

# --- Eğitim ---
def train():
    env = MazeEnv()
    net = SimpleNet()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    gamma = 0.95
    epsilon = 1.0
    min_epsilon = 0.05
    epsilon_decay = 0.995

    max_episodes = 10000
    max_steps_per_episode = 200

    total_steps = 0
    success_count = 0

    for episode in range(max_episodes):
        state = env.reset()
        for step in range(max_steps_per_episode):
            total_steps += 1

            q_values = net(state.unsqueeze(0))
            if random.random() < epsilon:
                action = random.randint(0,3)
            else:
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)

            # Güncelleme
            with torch.no_grad():
                next_q = net(next_state.unsqueeze(0)).max().item()
                target_q = reward + gamma * next_q * (0 if done else 1)

            pred_q = q_values[0, action]
            loss = criterion(pred_q, torch.tensor(target_q))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            if done:
                if reward > 0:
                    success_count += 1
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Terminal çıktısı
        success_rate = (success_count / (episode+1)) * 100
        print(f"\rEpisode: {episode+1} | Toplam Adım: {total_steps} | Başarı: {success_count} | Başarı Oranı: {success_rate:.3f}%  ", end='', flush=True)

        if (episode+1) % 100 == 0:
            print("\nMaze Son Durum:")
            env.render()
            print()

        if success_rate > 80.0 and episode > 100:
            print("\nBaşarı oranı %80'in üzerine çıktı. Eğitim tamamlandı.")
            break

    print("\nEğitim bitti.")

if __name__ == "__main__":
    train() burdaki nöron ağını yorumla nasıl çalışıyor kaç nöron var bilgiyi nasıl işliyor, impluse süreci nasıl, teknik detaylara gir
