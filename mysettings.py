class PacmanRewards:

    @staticmethod
    def perMove():
        return 1000

    @staticmethod
    def onEatingFood():
        return 100

    @staticmethod
    def onEatingGhost():
        return 200

    @staticmethod
    def onWin():
        return 500

    @staticmethod
    def onLoss():
        return -500


normalRuns = 10
trainingRuns = 2000

if __name__ == "__main__":
    from pacmanRL.run_helper import run

    run()
