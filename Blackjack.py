from random import Random
from dataclasses import dataclass, field
from enum import Enum
import copy
from tqdm import trange
import json
from itertools import chain, repeat
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import signal

stop = False

def handler(sig, frame):
    '''Handler for keyboard interrupt'''
    global stop
    stop = True
signal.signal(signal.SIGINT, handler)

random = Random()

@dataclass
class Card():
    name : str = field(default_factory=str)
    value : int = field(default_factory=int)

    def __repr__(self) -> str:
        name = self.name.upper()
        value = str(self.value) if self.value != 0 else str(11)
        return value + " " + name 

@dataclass
class State():
    card_sum    : int = field(default_factory=int)
    ace         : int = field(default_factory=int)
    dealer_card : int = field(default_factory=int)

    def get_total(self) -> int:
        '''Returns the best total from card_sum and ace'''
        totals = []
        # Starts from the largest sum and goes down
        for i in range(self.ace, -1, -1):
            total =  self.card_sum + i*11 + (self.ace - i)*1
            if total <= 21:
                return total
            totals.append(total)

        return min(totals)

    def __call__(self):
        return (self.card_sum, self.ace, self.dealer_card)

class Action(Enum):
    HIT = 0
    HOLD = 1

    def __invert__(self):
        return Action.HIT if self == Action.HOLD else Action.HOLD

    def __repr__(self) -> str:
        return "HIT" if self == Action.HIT else "HOLD"

class Deck():
    card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 12, 13, 14]
    card_types  = ['hearts', 'clubs', 'diamonds', 'spades']

    def __init__(self, explotation:bool) -> None:
        self.all_cards = self.init_deck()
        self.deck = self.shuffle()
        self.explotation = explotation

    def init_deck(self) -> list[Card]:
        return [Card(name, value) for name in self.card_types for value in self.card_values]

    def shuffle(self) -> iter:
        random.shuffle(self.all_cards)
        return iter(self.all_cards)

    def draw_card(self) -> int:
        card = next(self.deck, -1)

        if card == -1:
            self.deck = self.shuffle()
            card = next(self.deck)

        if self.explotation:
            print(f"{card}")

        if card.value >= 10: 
            return 10
        return card.value

class Player:

    def __init__(self, name:str, state=State()):
        self.state = state
        self.name = name
        self.holding = False

    def hold(self):
        self.holding = True

    def hit(self, new_card:int, ace:int):
        self.state.card_sum += new_card
        self.state.ace += int(ace)

    def get_total(self) -> int:
        return self.state.get_total()

    def busted(self) -> bool:
        if self.get_total() > 21:
            self.holding = True
            return True

        return False

class Environment:

    def __init__(self, deck, explotation:bool):
        self.deck = deck
        self.states = self.all_states()
        self.dealer = Player("Dealer", State())
        self.agent = Player("Agent", State())
        self.explotation = explotation

    def reset_env(self):
        self.dealer.state = State()
        self.dealer.holding = False

        self.agent.state = State()
        self.agent.holding = False        
        
        self.deck.shuffle()
        
    def initial_deal(self):
        self.hit(self.dealer)
        self.agent.state.dealer_card = self.dealer.get_total()

        self.hit(self.agent)
        self.hit(self.dealer)
        self.hit(self.agent)

        self.dealer.state.dealer_card = self.agent.get_total()

    def hit(self, player:Player):
        if self.explotation:
            print(f"{player.name} HIT: ", end='')

        card = self.deck.draw_card()

        # One in a million for a shuffle to occur in the middle of an episode
        # and for an ace to be drawn  after that resulting in as state that 
        # doesn't exist because the total would be larger than possible
        while card == 0 and player.state.ace == 4:
            card = self.deck.draw_card()

        ace = int(card == 0)

        player.hit(card, ace)

    def hold(self, player:Player):
        if self.explotation:
            print(f"{player.name} HOLD")
        player.hold()

    def all_states(self):
        states = []
                        
        for total in range(0, 22):
            for ace in range(0, 5):
                temp_state = State(total, ace)

                if 1 < temp_state.get_total() <= 21:
                    for dealer_total in range(2, 22):
                        state = copy.copy(temp_state)
                        state.dealer_card = dealer_total
                        states.append(state)
        
        return states

    def random_state(self):
        '''Used during exploration to expand the known observation space'''
        self.dealer.state = copy.copy(self.states[random.randint(0, len(self.states)-1)])
        self.agent.state  = copy.copy(self.states[random.randint(0, len(self.states))-1])

        if self.dealer.get_total() > 11:
            self.agent.state.dealer_card = random.randint(2, 11)
        else:
            self.agent.state.dealer_card = self.dealer.get_total()
        self.dealer.state.dealer_card = self.agent.get_total()

    def step(self, action:Action, player:Player) -> bool:
        ''' Performs one step with the given action for the specified player in an episode 
            Return value: True if HIT was played, False otherwise'''

        if not player.holding:
            if action == Action.HIT:
                self.hit(player)

                if player.busted():
                    return False

                return True
            else:
                self.hold(player)
        
        return False

    def evaluate_step(self, player:Player, opponent:Player) -> int:
        '''Returns the reward achieved by the last step'''
        gain = 1
        if not player.holding and not player.busted():
            gain = 0
        elif player.busted() or \
            (not opponent.busted() and opponent.get_total() > player.get_total()):
            gain = -1
        elif not opponent.busted() and \
            opponent.get_total() == player.get_total():
            gain = 0

        return gain

def random_policy() -> Action:
    '''Returns a random action'''
    if random.random() <= 0.5:
        return Action.HIT
    else:
        return Action.HOLD

def dealer_policy(state : State) -> Action:
    '''Returns an action based on a textbook dealer policy'''
    if state.get_total() < 17:
        return Action.HIT
    else:
        return Action.HOLD

def init_Q(state:State, Q:dict, value:callable):
    '''Initializes Q values'''
    if state not in Q:
        Q[state] = {action: value() for action in Action}

def NaiveMonteCarlo(episodes:list[list[State,Action,int]], gamma:float, policy:dict[State:Action]):
    ''' Every visit Naive Monte Carlo needs a whole episode in order to estimate the gain of an action, 
        since the mean of all occurences is used a large number of episodes is played in order 
        for the aproximation to be more accurate'''

    Q = {}

    for episode in episodes:
        for i, (state, action, _) in enumerate(episode):
            init_Q(state, Q, lambda: [])

            gamma_array = [[gamma**x] for x in range(len(episode)-i)]
            gain_array  = [gain for _, _, gain in episode[i:]]
            Q[state][action].append(np.dot(gain_array, gamma_array)[0])
    
    for state in Q:
        action = policy[state]
        if len(Q[state][action]):
            Q[state][action] = mean(Q[state][action]) 

    update_policy(Q, policy)

def IncrementalMonteCarlo(episode:list[State,Action,int], Q:dict[State,Action:int], gamma:float, alpha:float, policy:dict[State:Action]) -> list[State]:
    ''' Incremental Monte Carlo needs a whole episode to be played in order to estimate the gain of an action'''

    for i, (state, action, _) in enumerate(episode):
        gamma_array = [[gamma**x] for x in range(len(episode)-i)]
        gain_array  = [gain for _, _, gain in episode[i:]]
        Q_aprox = np.dot(gain_array, gamma_array)[0]
        Q[state][action] += alpha*(Q_aprox - Q[state][action])

    Q_temp = {state : Q[state] for state, _, _ in episode}
    update_policy(Q_temp, policy)

def Q_learning(Q:dict[State,Action:int], state:State, action:Action, new_state:State, gain:int, gamma:float, alpha:float, policy:dict[State:Action]) \
    -> Action:
    '''Updates the policy with respect to optimal Q-values in current state'''

    if new_state is not None:
        max_action = max(Q[new_state], key=Q[new_state].get)
        Q_aprox = gain + gamma*Q[new_state][max_action]
    else:
        Q_aprox = gain

    Q[state][action] += alpha*(Q_aprox - Q[state][action])

    # Da li odmah obnoviti politiku ili tek nakon vise partija - unosi vece kasnjenje 
    Q_temp = {state : Q[state]}
    update_policy(Q_temp, policy)

    return policy[new_state] if new_state is not None else None

def SARSA(Q:dict[State,Action:int], state:State, action:Action, new_state:State, new_action:Action, gain:int, gamma:float, alpha:float, policy:dict[State:Action]):
    '''Updates the policy with respect to actual Q-values in current state'''
 
    if new_state is not None:
        Q_aprox = gain + gamma*Q[new_state][new_action]
    else:
        Q_aprox = gain
    
    Q[state][action] += alpha*(Q_aprox - Q[state][action])

    # Da li odmah obnoviti politiku ili tek nakon vise partija 
    Q_temp = {state : Q[state]}
    update_policy(Q_temp, policy)

    return new_action

def update_policy(Q:dict[State,Action:int], policy:dict[State:Action]):
    ''' Finds the greedy policy with respect to the passed Q-values '''
    for state in Q.keys():
        policy[state] = max(Q[state], \
            key=lambda k: Q[state].get(k) if not isinstance(Q[state].get(k), list) else 2*random.random()-1)

def empty_func(*args):
    ''' Placeholder for learning methods that are not currently in use'''
    return None

if __name__ == "__main__":

    deck = Deck(explotation=False)
    env = Environment(deck, explotation=False)

    methods = {                    
        "Naive Monte Carlo"       : ( 5000,   1, 0.5, 500), 
        "Incremental Monte Carlo" : ( 5000, 0.9, 0.5, 500),
        "Q learning"              : ( 1000, 0.9, 0.75, 500),
        "SARSA"                   : ( 1000, 0.9, 0.5, 500),
        "No learning"             : (    0,   0,   0, 0)
    }

    method_list = []
    for i, method in enumerate(methods.keys()):
        print(f"{str(i+1)}. {method}")
        method_list.append(method)

    prompts = chain(["Choose a method of learning for the agent: "], repeat(f"Method does not exist\nChoose another method: "))
    replies = map(input, prompts)
    method = next(filter(lambda s: (s in methods.keys() and s != "No learning") or int(s) in range(1,len(methods.keys())), replies))
    if method.isnumeric():
        method = method_list[int(method)-1]

    executable = {
        "Naive Monte Carlo"       : NaiveMonteCarlo if method == "Naive Monte Carlo" else empty_func, 
        "Incremental Monte Carlo" : IncrementalMonteCarlo if method == "Incremental Monte Carlo" else empty_func,
        "Q learning"              : Q_learning if method == "Q learning" else empty_func,
        "SARSA"                   : SARSA if method == "SARSA" else empty_func,
        "No learning"             : empty_func
    }

    prompts = chain(["Choose a method of learning for the dealer: "], repeat(f"Method does not exist\nChoose another method: "))
    replies = map(input, prompts)
    dealer_method = next(filter(lambda s: s in methods.keys() or int(s) in range(1,len(methods.keys())+1), replies))
    if dealer_method.isnumeric():
        dealer_method = method_list[int(dealer_method)-1]

    d_executable = {
        "Naive Monte Carlo"       : NaiveMonteCarlo if dealer_method == "Naive Monte Carlo" else empty_func, 
        "Incremental Monte Carlo" : IncrementalMonteCarlo if dealer_method == "Incremental Monte Carlo" else empty_func,
        "Q learning"              : Q_learning if dealer_method == "Q learning" else empty_func,
        "SARSA"                   : SARSA if dealer_method == "SARSA" else empty_func,
        "No learning"             : None
    }

    # iterations     - number of iterations per episode
    # alpha          - filter value
    # gamma          - forgetting factor
    # step           - number of future steps to take into account when forming an aproximation for Q in SARSA
    iterations, alpha, gamma, max_iterations = methods[method]

    policy = {state() : random_policy() for state in env.states if state.dealer_card <= 11}
    dealer_policy = {state() : dealer_policy(state) for state in env.states}
    
    last_policy = copy.deepcopy(policy)
    last_dealer_policy = copy.deepcopy(dealer_policy)

    best_policy = copy.deepcopy(policy)
    best_results = []
    best_result = -iterations
    num_of_iter = 0

    Q = {}
    if method != "Naive Monte Carlo":
        Q = {state() : {action: 2*random.random()-1 for action in Action} for state in env.states if state.dealer_card <= 11}
    
    d_Q = {}
    if dealer_method not in  ["No learning", "Naive Monte Carlo"]:
        d_Q = {state() : {action: 2*random.random()-1 for action in Action} for state in env.states}

    plt.title(f"Sum of all results made in {iterations} episodes")

    try:
        for k in range(1, max_iterations+1):
            if stop:
                break

            episodes = []
            dealer_episodes = []
            results = []

            # Simulation of iteration episodes
            for _ in range(iterations):
                episode = []
                d_episode = []

                env.reset_env()
                env.random_state()

                state = env.agent.state()
                # Deljenje karata unosi nasumicnost u sve partije, da li je potrebno birati prvu akciju nausmicno
                action = policy[state] # if random.random() > 0.75 else random.choice([a for a in Action])
                while env.step(action, env.agent):
                    gain = env.evaluate_step(env.agent, env.dealer)
                    episode.append((state, action, gain))

                    new_state = env.agent.state()
                    new_action = policy[new_state]

                    action_q = executable["Q learning"](Q, state, action, new_state, gain, gamma, alpha, policy)
                    action_s = executable["SARSA"](Q, state, action, new_state, new_action, gain, gamma, alpha, policy)

                    action = action_q or action_s or new_action
                    state = new_state

                if not env.agent.busted():
                    d_state = env.dealer.state()
                    d_action = dealer_policy[d_state]
                    while env.step(d_action, env.dealer):
                        gain = env.evaluate_step(env.dealer, env.agent)
                        d_episode.append((d_state, d_action, gain))

                        new_d_state = env.dealer.state()
                        new_d_action = dealer_policy[new_d_state]

                        d_action_q = d_executable["Q learning"](d_Q, d_state, d_action, new_d_state, \
                            gain, gamma, alpha, dealer_policy)
                        d_action_s = d_executable["SARSA"](d_Q, d_state, d_action, new_d_state, new_d_action, \
                            gain, gamma, alpha, dealer_policy)
                        
                        d_action = d_action_q or d_action_s or new_d_action
                        d_state = new_d_state

                    gain = env.evaluate_step(env.dealer, env.agent)
                    d_episode.append((d_state, d_action, gain))

                    d_executable["Q learning"](d_Q, d_state, d_action, None, gain, gamma, alpha, dealer_policy)
                    d_executable["SARSA"](d_Q, d_state, d_action, None, None, gain, gamma, alpha, dealer_policy)
                else:
                    env.dealer.hold()

                gain = env.evaluate_step(env.agent, env.dealer)
                episode.append((state, action, gain))

                executable["Q learning"](Q, state, action, None, gain, gamma, alpha, policy)
                executable["SARSA"](Q, state, action, None, None, gain, gamma, alpha, policy)
                
                executable["Incremental Monte Carlo"](episode, Q, gamma, alpha, policy)
                d_executable["Incremental Monte Carlo"](d_episode, d_Q, gamma, alpha, dealer_policy)

                episodes.append(episode)

                if len(d_episode):
                    dealer_episodes.append(d_episode)

                results.append(episode[-1])

            sum_results = sum([gain for _, _, gain in results])

            executable["Naive Monte Carlo"](episodes, gamma, policy)
            d_executable["Naive Monte Carlo"](dealer_episodes, gamma, dealer_policy)

            plt.scatter(k, sum_results, marker="o")
            plt.pause(0.0001)

            if sum_results > best_result:
                best_results = copy.deepcopy(results)
                best_result = sum_results
                best_policy = copy.deepcopy(policy)
                num_of_iter = k

            if last_policy == policy and last_dealer_policy == dealer_policy: 
                print(f"Optimal policy reached in {k} iterations")
                break

            last_policy = copy.deepcopy(policy)
            last_dealer_policy = copy.deepcopy(dealer_policy)

    except Exception as e: 
        print(e)
    
    # When both the agent and dealer are learning their policies, the best agent policy is the last one
    if dealer_method != "No learning":
        best_results = results
        best_result = sum_results
        best_policy = policy
        num_of_iter = k

    print(f"\nExiting after {k} iteration")
    input("Continue ")
    plt.close()

    plt.show()
    result_values = [gain for _, _, gain in best_results]

    won  = round(result_values.count(1)/iterations * 100)
    lost = round(result_values.count(-1)/iterations * 100)
    drawn= round(result_values.count(0)/iterations * 100)
    print(f"Won games: {won}%")
    print(f"Lost games: {lost}%")
    print(f"Drawn games: {drawn}%")

    out_policy = {str(state):repr(action) for state, action in best_policy.items()}
    out_dealer_policy = {str(state):repr(action) for state, action in dealer_policy.items()}

    results = [''.join((str(state),", ", repr(action), ", ", str(gain))) for state, action, gain in best_results]

    out = {
        "episode_iterations" : iterations,
        "algorithm" : method,
        "dealer_algorithm" : dealer_method,
        "won" : won,
        "lost" : lost,
        "iterations" : num_of_iter,
        "result": best_result,
        "result_list" : results,
        "policy" : out_policy,
        "dealer_policy" : out_dealer_policy
    }

    filename = input("Enter filename: ")

    json_obj = json.dumps(out, indent=2)
    with open(filename + ".json", "w+") as outfile:
        outfile.write(json_obj)

    
