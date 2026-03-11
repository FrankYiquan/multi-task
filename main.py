from config import get_args
import train


if __name__ == "__main__":

    args = get_args()
    train.train(args)