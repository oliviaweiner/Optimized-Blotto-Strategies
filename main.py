from hardcoded_strategy import create_hardcoded_agent
import blotto_game

def main():
    agent1 = create_hardcoded_agent([[1,3,3,1]], 4, 8)
    agent2 = create_hardcoded_agent([[2,2,2,2]], 4, 8)
    agent3 = create_hardcoded_agent([[0,1,3,4]], 4, 8)
    agent4 = create_hardcoded_agent([[0,0,4,4],[0,2,3,3]], 4, 8, [.25, .75])
    print(blotto_game.simulate_blotto_game([agent1, agent2, agent3, agent4], 10))

if __name__ == '__main__':
    main()
