import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from collections import deque

#Original size = (400,600,3)

# ==================================================
# ============= PREPROCESSING FUNCTION =============
# ==================================================

def preprocess_image(img):
    """
    Do a preprocessing of an image of CartPole.
    
    Parameter
    --------------------
    img: ndarray
        an image of CartPole.

    Return
    --------------------
    img_preprcs: ndarray
        img preprocessed.
    """

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    final_img = cv2.resize(gray_img, (84,84), interpolation=cv2.INTER_AREA)
    return final_img

# ==================================================
# =============== TEST PREPROCESSING =============== 
# ==================================================

def test_preprocess_cartpole(preprocess_fun, n_episodes):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    for _ in range(1, n_episodes+1):
        env.reset()
        done = False
        while not done:
            curr_img = env.render()
            cv2.imshow("CartPole", preprocess_fun(curr_img))
            cv2.waitKey(0)

            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated

    cv2.destroyAllWindows()
    env.close()

# ==================================================
# ================ TRAINING FUNCTION ===============
# ==================================================

def initialize_observations(last_n_frames):
    obs_stckd = deque(maxlen=last_n_frames)
    next_obs_stckd = deque(maxlen=last_n_frames)

    for _ in range(last_n_frames):
        obs_stckd.append(np.zeros((84,84), dtype=np.uint8))
        next_obs_stckd.append(np.zeros((84,84), dtype=np.uint8))

    return obs_stckd, next_obs_stckd

def train_cartpole(agent, target_total_frames, preprocess_fun=preprocess_image):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    #Observation stacked are generated.
    obs_stckd = deque(maxlen=agent.last_n_frames)
    next_obs_stckd = deque(maxlen=agent.last_n_frames)

    for _ in range(agent.last_n_frames):
        obs_stckd.append(np.zeros((84,84), dtype=np.uint8))
        next_obs_stckd.append(np.zeros((84,84), dtype=np.uint8))

    scores = []
    avg_scores = []
    q_values = deque(maxlen=10000)
    total_frames = 0
    episode = 1
    while total_frames <= target_total_frames:
        #Current episode infos are initialized.
        env.reset()
        # obs_stckd, next_obs_stckd = initialize_observations(agent.last_n_frames)

        init_img = preprocess_fun(env.render())
        obs_stckd.append(init_img)
        next_obs_stckd.append(init_img)
        frames = 0
        score = 0
        done = False
        while not done:
            #Agent chooses an action.
            action, q = agent.choose_action(obs_stckd)
            q_values.append(q)

            #Agent performs action choosen.
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            #Add image.
            next_img = preprocess_fun(env.render())
            next_obs_stckd.append(next_img)

            #Do training step.
            agent.store(obs_stckd, action, reward, next_obs_stckd, done)
            agent.train()

            #Update infos
            obs_stckd.append(next_img)
            score += reward
            frames += 1

            #Update target net.
            if (total_frames + frames) % agent.update_rate_target == 0:
                agent.update_target()

            if (total_frames + frames) % 25000 == 0:
                agent.save_model("CartPole_DDQN_{}.pth".format(total_frames + frames))

        #Update stats.
        total_frames += frames
        scores.append(score)
        avg_scores.append(np.mean(scores[-100:]))

        #Print stats.
        print('- episode: {} ; score: {}; avg scores: {:.2f}; avg q: {:.2f}; frames: {}; total frames: {}; epsilon: {:.2f}'.format(episode, score, avg_scores[-1], np.mean(q_values), frames, total_frames, agent.epsilon))

        #Next episode.
        episode += 1
    
    agent.save_model("CartPole_DDQN.pth")

    env.close()


# ==================================================
# ================ RUNNING FUNCTION ================
# ==================================================
    
def run_cartpole(agent, n_episodes, preprocess_fun=preprocess_image):
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    #Observation stacked are generated.
    obs_stckd = deque(maxlen=agent.last_n_frames)
    next_obs_stckd = deque(maxlen=agent.last_n_frames)

    for _ in range(agent.last_n_frames):
        obs_stckd.append(np.zeros((84,84), dtype=np.uint8))
        next_obs_stckd.append(np.zeros((84,84), dtype=np.uint8))

    #Agent is run for CartPole.
    scores = []
    for episode in range(1, n_episodes+1):
        env.reset()
        #obs_stckd, next_obs_stckd = initialize_observations(agent.last_n_frames)

        init_img = preprocess_fun(env.render())
        obs_stckd.append(init_img)
        next_obs_stckd.append(init_img)
        score = 0
        done = False
        while not done:
            curr_img = env.render()
            cv2.imshow("CartPole", cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            #Agent chooses an action.
            action, _ = agent.choose_action(obs_stckd)

            #Agent performs action choosen.
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            #Add image.
            next_img = preprocess_fun(env.render())
            next_obs_stckd.append(next_img)

            #Update infos
            obs_stckd.append(next_img)
            score += reward

        #Add score.
        scores.append(score)

        #Print stats.
        print('- episode: {} ; score: {}; avg score: {:.2f}; std score: {:.2f}'.format(episode, score, np.mean(scores), np.std(scores)))

    cv2.destroyAllWindows()
    env.close()


# ==================================================
# ====================== MAIN ======================
# ==================================================

is_preprocessing_tested = False
is_trained = False

if is_preprocessing_tested:
    test_preprocess_cartpole(preprocess_image, 50)
elif is_trained:
    agent = Agent(15000, 32, 1000, lr=10**-4, eps_decay=3.96*10**-5)
    target_total_frames = 200000

    train_cartpole(agent, target_total_frames)
else:
    agent = Agent(1, 1, 1, is_trained=False)
    agent.load_model("models/final_version/CartPole_DDQN.pth")
    n_episodes = 100

    run_cartpole(agent, n_episodes)