from Blackjack import Environment, Deck, Action
import json
from itertools import chain, repeat

def continue_question(msg : str) -> bool:
    prompts = chain([msg], \
                    repeat(f"Only 'y' or 'n' are valid responses. \n{msg}"))
    replies = map(input, prompts)

    valid = next(filter(lambda s: s == 'y' or s == 'n', replies))

    print("------------------------------")

    if valid == 'y':
        return True
    return False

deck = Deck(explotation=True)
env = Environment(deck, explotation=True)

file = input("Name of file with policy: ")
file += ".json"

try:
    with open(file) as in_file:
        policy_dict = json.load(in_file)
        policy = {eval(state): Action.HIT if action == "HIT" else Action.HOLD for state, action in policy_dict["policy"].items()}
        dealer_policy = {eval(state):Action.HIT if action == "HIT" else Action.HOLD for state, action in policy_dict["dealer_policy"].items()}

        while continue_question("Play game? "):
            episode = []
            env.reset_env()
            env.initial_deal()

            print(f"Initial agent state: {env.agent.state()}")
            print(f"Initial dealer state: {env.dealer.state()}")
            print("-------------------------------------")


            state = env.agent.state()
            action = policy[state]
            while env.step(action, env.agent):
                gain = env.evaluate_step(env.agent, env.dealer)
                episode.append((state, action, gain))

                state = env.agent.state()
                action = policy[state]

            if not env.agent.busted():
                d_state = env.dealer.state()
                d_action = dealer_policy[d_state]
                while env.step(d_action, env.dealer):
                    d_state = env.dealer.state()
                    d_action = dealer_policy[d_state]
            else:
                env.dealer.hold()

            gain = env.evaluate_step(env.agent, env.dealer)
            episode.append((state, action, gain))

            print(f"Final agent state: {env.agent.state()}")
            print(f"Final dealer state: {env.dealer.state()}")
            print("-------------------------------------")

            if gain == 0:
                print("DRAW")
            elif gain == 1:
                print("WIN")
            elif env.agent.busted():
                print("BUSTED")
            elif gain == -1:
                print("LOSS")
                
except KeyboardInterrupt:
    print("\nExiting")
