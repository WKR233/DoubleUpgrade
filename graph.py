import numpy as np
import matplotlib.pyplot as plt
import rl_utils as utils
# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asarray(data, float)

def reward_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[:].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asarray(data, float)

if __name__ == "__main__":

        train_loss_path = r"./train_loss.txt"   # 存储文件路径
        
        
        y_train_loss = data_read(train_loss_path)        # loss值，即y轴
        x_train_loss = range(len(y_train_loss))			 # loss的数量，即x轴

	


        plt.figure(1)
        # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
        # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
        plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
        plt.xlabel('iters')    # x轴标签
        plt.ylabel('loss')     # y轴标签
        plt.legend()
        plt.title('Loss curve')
        plt.show()
        plt.savefig("loss.png")

        mv_loss = utils.moving_average(y_train_loss, 99)
        plt.plot(x_train_loss, mv_loss)
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.title('Loss curve')
        plt.savefig("loss_smoothed.png")

        for i in range (4):
            
            reward_path = r"./rewardActor-"+str(i)+".txt"
            y_reward = reward_read(reward_path)
            x_reward = range(len(y_reward))	

            plt.figure(i+2)
            mv_reward = utils.moving_average(y_reward, 99)
            plt.plot(x_reward, y_reward, linewidth=1, linestyle="solid", label="reward")
            plt.plot(x_reward, mv_reward)
            plt.legend()
            plt.xlabel('Iters')
            plt.ylabel('Reward')
            plt.title('Reward curve')
            plt.savefig("reward_smoothed_actor_"+str(i)+".png")
