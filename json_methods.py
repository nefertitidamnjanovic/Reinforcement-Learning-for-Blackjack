import json
import matplotlib.pyplot as plt
import numpy as np
import copy
from Blackjack import Deck, State

def plot_policy(ax, policy:dict, method:str, params):
    ax.set_title(f"{method}: iterations - {params[0]}, iterations per episode - {params[1]}, result - {params[2]}")
    colors = {
        "HIT" : "red",
        "HOLD": "blue"
    }

    max_y = 11

    for state, action in policy.items():
        card_sum, ace, dealer_card = eval(state)
        state = State(card_sum, ace, dealer_card)
        total = state.get_total()

        if dealer_card > max_y:
            max_y = dealer_card

        ax.scatter(total, dealer_card, marker="s", color=colors[action], s=800)
    
    ax.set_xticks(range(1, 22))
    ax.set_yticks(range(1, max_y+1))
    ax.set_xlabel("Agent total")
    ax.set_ylabel("Dealer card")
    ax.legend(["HIT", "HOLD"])

    leg = ax.get_legend()
    leg.legend_handles[0]._sizes = [30]
    leg.legend_handles[1]._sizes = [30]
    leg.legend_handles[0].set_color('red')
    leg.legend_handles[1].set_color('blue')

    return ax

def plot_results(ax, results):
    totals_won = [0]*22
    totals_drawn = [0]*22
    totals_lost = [0]*22
    
    for result in results:
        result = result.replace("(","").replace(")","").split(",")
        card_sum = int(result[0])
        ace = int(result[1])
        dealer_card = int(result[2])
        action = result[3]
        gain = int(result[4])

        state =  State(card_sum, ace, dealer_card)
        total = state.get_total()

        if gain == 1:
            totals_won[int(total)] += 1
        elif gain == -1:
            totals_lost[int(total)] += 1
        else:
            totals_drawn[int(total)] += 1

    min_totals_won = min([i for i in range(1,22) if totals_won[i] > 0])
    min_totals_lost = min([i for i in range(1,22) if totals_lost[i] > 0])
    min_totals_draw = min([i for i in range(1,22) if totals_drawn[i] > 0])

    min_totals = min([min_totals_lost, min_totals_won, min_totals_draw])

    for i in range(min_totals,22):
        ax.bar(i-0.33, totals_won[i], color="red", width=0.33)
        ax.bar(i, totals_lost[i], color="blue", width=0.33)
        ax.bar(i+0.33, totals_drawn[i], color="green", width=0.33)

    ax.set_xticks(range(min_totals, 22))
    ax.set_xlabel("Agent total")
    ax.set_ylabel("Win/Loss/Draw count")
    ax.legend(["Won", "Lost", "Drawn"])

    leg = ax.get_legend()
    leg.legend_handles[0].set_color('red')
    leg.legend_handles[1].set_color('blue')
    leg.legend_handles[2].set_color('green')

    return ax

def load_policy():
    while 1:
        try:
            filename = input("Enter policy file name: ")
            with open(filename, "r") as in_file:
                policy_dict = json.load(in_file)
                params = (policy_dict["iterations"], policy_dict["episode_iterations"])

                return policy_dict["policy"], params
        except Exception as e:
            print(e)

def all_states():
    states = []

    all_totals = [total for total in range(2, 22)]

    for total in all_totals:
        for ace in range(0, 5):
            temp_state = State(total, ace)

            if temp_state.get_total() <= 21:
                states.append(temp_state)

    print(states)

if __name__ == "__main__":
    file = input("Name of file: ")
    file += ".json"

    try:
        with open(file) as in_file:
            policy_dict = json.load(in_file)

            fig, ax = plt.subplots(1,2)
            params = (policy_dict["iterations"], policy_dict["episode_iterations"], policy_dict["result"])
            plot_policy(ax[0], policy_dict["policy"], policy_dict["algorithm"], params)
            plot_results(ax[1], policy_dict["result_list"])

            if "dealer_policy" in policy_dict:
                fig1, ax1 = plt.subplots(1,1)
                params = (policy_dict["iterations"], policy_dict["episode_iterations"], -policy_dict["result"])
                plot_policy(ax1, policy_dict["dealer_policy"], policy_dict["dealer_algorithm"], params)

            print(f"Won : {policy_dict['won']}%")
            print(f"Lost: {policy_dict['lost']}%")

            plt.show()
    except KeyboardInterrupt:
        print("\nExiting")
        plt.close()