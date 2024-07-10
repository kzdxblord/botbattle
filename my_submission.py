from collections import defaultdict, deque
import random
from typing import Optional, Tuple, Union, cast
from risk_helper.game import Game
from risk_shared.models.card_model import CardModel
from risk_shared.queries.query_attack import QueryAttack
from risk_shared.queries.query_claim_territory import QueryClaimTerritory
from risk_shared.queries.query_defend import QueryDefend
from risk_shared.queries.query_distribute_troops import QueryDistributeTroops
from risk_shared.queries.query_fortify import QueryFortify
from risk_shared.queries.query_place_initial_troop import QueryPlaceInitialTroop
from risk_shared.queries.query_redeem_cards import QueryRedeemCards
from risk_shared.queries.query_troops_after_attack import QueryTroopsAfterAttack
from risk_shared.queries.query_type import QueryType
from risk_shared.records.moves.move_attack import MoveAttack
from risk_shared.records.moves.move_attack_pass import MoveAttackPass
from risk_shared.records.moves.move_claim_territory import MoveClaimTerritory
from risk_shared.records.moves.move_defend import MoveDefend
from risk_shared.records.moves.move_distribute_troops import MoveDistributeTroops
from risk_shared.records.moves.move_fortify import MoveFortify
from risk_shared.records.moves.move_fortify_pass import MoveFortifyPass
from risk_shared.records.moves.move_place_initial_troop import MovePlaceInitialTroop
from risk_shared.records.moves.move_redeem_cards import MoveRedeemCards
from risk_shared.records.moves.move_troops_after_attack import MoveTroopsAfterAttack
from risk_shared.records.record_attack import RecordAttack
from risk_shared.records.types.move_type import MoveType


# We will store our enemy in the bot state.
class BotState():
    def __init__(self):
        self.enemy: Optional[int] = None


def main():
    
    # Get the game object, which will connect you to the engine and
    # track the state of the game.
    game = Game()
    bot_state = BotState()
   
    # Respond to the engine's queries with your moves.
    while True:

        # Get the engine's query (this will block until you receive a query).
        query = game.get_next_query()

        # Based on the type of query, respond with the correct move.
        def choose_move(query: QueryType) -> MoveType:
            match query:
                case QueryClaimTerritory() as q:
                    return handle_claim_territory(game, bot_state, q)

                case QueryPlaceInitialTroop() as q:
                    return handle_place_initial_troop(game, bot_state, q)

                case QueryRedeemCards() as q:
                    return handle_redeem_cards(game, bot_state, q)

                case QueryDistributeTroops() as q:
                    return handle_distribute_troops(game, bot_state, q)

                case QueryAttack() as q:
                    return handle_attack(game, bot_state, q)

                case QueryTroopsAfterAttack() as q:
                    return handle_troops_after_attack(game, bot_state, q)

                case QueryDefend() as q:
                    return handle_defend(game, bot_state, q)

                case QueryFortify() as q:
                    return handle_fortify(game, bot_state, q)
        
        # Send the move to the engine.
        game.send_move(choose_move(query))
                

def handle_claim_territory(game: Game, bot_state: BotState, query: QueryClaimTerritory) -> MoveClaimTerritory:
    """At the start of the game, you can claim a single unclaimed territory every turn 
    until all the territories have been claimed by players."""
    
    continents = {
        'NA': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'SA': [28, 29, 30, 31],
        'EU': [9, 10, 11, 12, 13, 14, 15],
        'AS': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        'AU': [38, 39, 40, 41],
        'AF': [32, 33, 34, 35, 36, 37]
    }

    

    good_territories = {
        'NA': [6,1,8,3],
        'SA': [29,31],
        'AU': [40],
        'AF': [33]
    }

    unclaimed_territories = game.state.get_territories_owned_by(None)
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    claimed_territories = set(range(41)) - set(unclaimed_territories)
    my_continents = set()
    for continent, territories in continents.items():
        if set(territories) & set(my_territories):
            my_continents.add(continent)

    # We will try to always pick new territories that are next to ones that we own,
    # or a random one if that isn't possible.
    adjacent_territories = game.state.get_all_adjacent_territories(my_territories)

    # We can only pick from territories that are unclaimed and adjacent to us.
    available = list(set(unclaimed_territories) & set(adjacent_territories))
    prioritized_available = [t for t in available if any(t in continents[c] for c in my_continents)]

    if len(prioritized_available) != 0:
        available = prioritized_available
    if len(available) != 0:

        # We will pick the one with the most connections to our territories
        # this should make our territories clustered together a little bit.
        def count_adjacent_friendly(x: int) -> int:
            return len(set(my_territories) & set(game.state.map.get_adjacent_to(x)))

        selected_territory = sorted(available, key=lambda x: count_adjacent_friendly(x), reverse=True)[0]
    
    # pick territories with the greatest strategic advantage
    else:
        na_claimed = any(territory in continents['NA'] for territory in claimed_territories)
        sa_claimed = any(territory in continents['SA'] for territory in claimed_territories)
        au_claimed = any(territory in continents['AU'] for territory in claimed_territories)
        af_claimed = any(territory in continents['AF'] for territory in claimed_territories)

        if not na_claimed:
            selected_territory = good_territories['NA'][random.randint(0, 3)]
        elif not sa_claimed:
            selected_territory = good_territories['SA'][random.randint(0,1)]
        elif not au_claimed:
            selected_territory = good_territories['AU'][0]
        elif not af_claimed:
            selected_territory = good_territories['AF'][0]
        else:
            selected_territory = sorted(unclaimed_territories, key=lambda x: len(game.state.map.get_adjacent_to(x)), reverse=True)[0]
         # Or if there are no such territories, we will pick just an unclaimed one with the greatest degree.
    return game.move_claim_territory(query, selected_territory)


def handle_place_initial_troop(game: Game, bot_state: BotState, query: QueryPlaceInitialTroop) -> MovePlaceInitialTroop:
    """After all the territories have been claimed, you can place a single troop on one
    of your territories each turn until each player runs out of troops."""
    
    # We will place troops along the territories on our border.
    border_territories = game.state.get_all_border_territories(
        game.state.get_territories_owned_by(game.state.me.player_id)
    )

    # Calculate threat levels for all border territories
    threat_levels = {territory: get_threat_territory(game, bot_state, territory) for territory in border_territories}

    # Calculate priority scores for each border territory
    priority_scores = {}
    for territory in border_territories:
        troops = game.state.territories[territory].troops
        threat = threat_levels[territory]
        priority_scores[territory] = threat / (troops + 1)  # Adding 1 to avoid division by zero

    # Find the territory with the highest priority score
    max_priority_territory = max(priority_scores.items(), key=lambda item: item[1])[0]

    return game.move_place_initial_troop(query, max_priority_territory)


def handle_redeem_cards(game: Game, bot_state: BotState, query: QueryRedeemCards) -> MoveRedeemCards:
    """Redeem cards in the optimal way to maximize troops and strategic advantage."""


    card_sets: list[Tuple[CardModel, CardModel, CardModel]] = []
    cards_remaining = game.state.me.cards.copy()

    while len(cards_remaining) >= 5:
        card_set = game.state.get_card_set(cards_remaining)
        assert card_set is not None
        card_sets.append(card_set)
        cards_remaining = [card for card in cards_remaining if card not in card_set]


    if game.state.card_sets_redeemed > 16 and query.cause == "turn_started":
        card_set = game.state.get_card_set(cards_remaining)
        while card_set is not None:
            card_sets.append(card_set)
            cards_remaining = [card for card in cards_remaining if card not in card_set]
            card_set = game.state.get_card_set(cards_remaining)


    territory_card_sets = [
        set for set in card_sets if any(card.territory_id in game.state.get_territories_owned_by(game.state.me.player_id) for card in set)
    ]
    if territory_card_sets:
        card_sets = territory_card_sets + [set for set in card_sets if set not in territory_card_sets]

    return game.move_redeem_cards(query, [(x[0].card_id, x[1].card_id, x[2].card_id) for x in card_sets])


def scale(value, input_min=0, input_max=100, output_min=0, output_max=50):
    if value > input_max:
        return input_max
    scaled_value = output_min + (value - input_min) * (output_max - output_min) / (input_max - input_min)
    return scaled_value

def get_threat_territory(game: Game, bot_state: BotState, territory:int)-> float:

    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)

    continent_bonuses = {
        'AS': 7,
        'NA': 5,
        'EU': 5,
        'AF': 3,
        'SA': 2,
        'AU': 2
    }

    continents = {
        'NA': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'SA': [28, 29, 30, 31],
        'EU': [9, 10, 11, 12, 13, 14, 15],
        'AS': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        'AU': [38, 39, 40, 41],
        'AF': [32, 33, 34, 35, 36, 37]
    }

    territory_to_cont = {0: 'NA', 1: 'NA', 2: 'NA', 3: 'NA', 4: 'NA', 5: 'NA', 6: 'NA', 7: 'NA', 8: 'NA',
    9: 'EU', 10: 'EU', 11: 'EU', 12: 'EU', 13: 'EU', 14: 'EU', 15: 'EU', 16: 'AS', 17: 'AS', 18: 'AS', 19: 'AS', 20: 'AS',
    21: 'AS', 22: 'AS', 23: 'AS', 24: 'AS', 25: 'AS', 26: 'AS', 27: 'AS', 28: 'SA', 29: 'SA', 30: 'SA', 31: 'SA', 32: 'AF', 33: 'AF', 34: 'AF',
    35: 'AF', 36: 'AF', 37: 'AF', 38: 'AU', 39: 'AU', 40: 'AU', 41: 'AU'
}
    chokepoints = [0,21,4,10,2,30,29,36,34,22,24,40,15,36]

    army_strength = game.state.territories[territory].troops #50
    adjacent_strength = 0 #30
    adjacent_territories = game.state.get_all_adjacent_territories([territory]) #
    continent_importance = continent_bonuses[territory_to_cont[territory]] #10
    strategic_position = 0 #10

    for i in adjacent_territories:
        if i not in my_territories:
            strength = game.state.territories[i].troops
            adjacent_strength += strength
    
    if territory in chokepoints:
        strategic_position = 10

    army_strength = scale(army_strength)
    adjacent_strength = scale(adjacent_strength, input_min=0, input_max=100, output_min=0, output_max=30)
    continent_importance = scale(continent_importance, input_min=0, input_max=7, output_min=0, output_max=10)

    threat = adjacent_strength + continent_importance + strategic_position - army_strength

    return threat / 50


def handle_distribute_troops(game: Game, bot_state: BotState, query: QueryDistributeTroops) -> MoveDistributeTroops:
    """After you redeem cards (you may have chosen to not redeem any), you need to distribute
    all the troops you have available across your territories. This can happen at the start of
    your turn or after killing another player.
    """
    continents = {
        'NA': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'SA': [28, 29, 30, 31],
        'EU': [9, 10, 11, 12, 13, 14, 15],
        'AS': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        'AU': [38, 39, 40, 41],
        'AF': [32, 33, 34, 35, 36, 37]
    }

    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    total_troops = game.state.me.troops_remaining
    distributions = defaultdict(lambda: 0)
    border_territories = game.state.get_all_border_territories(my_territories)

    # We neeed to remember we have to place our matching territory bonus if we have one.
    if len(game.state.me.must_place_territory_bonus) != 0:
        assert total_troops >= 2
        distributions[game.state.me.must_place_territory_bonus[0]] += 2
        total_troops -= 2

    # Calculate threat levels for all border territories
    threat_levels = {territory: get_threat_territory(game, bot_state, territory) for territory in border_territories}

    # Sort territories by threat leveel in descending order
    sorted_territories = sorted(threat_levels.items(), key=lambda x: x[1], reverse=True)

    # Allocate troops to the top 3 most threatened territories
    top_territories = sorted_territories[:3]
    for territory, threat in top_territories:
        if total_troops <= 0:
            break
        distributions[territory] += total_troops // len(top_territories)
    
    # Allocate any remaining troops
    remaining_troops = total_troops % len(top_territories)
    for i in range(remaining_troops):
        distributions[top_territories[i][0]] += 1

    return game.move_distribute_troops(query, distributions)

def handle_attack(game: Game, bot_state: BotState, query: QueryAttack) -> Union[MoveAttack, MoveAttackPass]:
    """After the troop phase of your turn, you may attack any number of times until you decide to
    stop attacking (by passing). After a successful attack, you may move troops into the conquered
    territory. If you eliminated a player you will get a move to redeem cards and then distribute troops."""

    # We will attack someone.
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    bordering_territories = game.state.get_all_adjacent_territories(my_territories)

    def attack_weakest(territories: list[int]) -> Optional[MoveAttack]:
        # We will attack the weakest territory from the list.
        territories = sorted(territories, key=lambda x: game.state.territories[x].troops)
        for candidate_target in territories:
            candidate_attackers = sorted(list(set(game.state.map.get_adjacent_to(candidate_target)) & set(my_territories)), key=lambda x: game.state.territories[x].troops, reverse=True)
            for candidate_attacker in candidate_attackers:
                if game.state.territories[candidate_attacker].troops > 1:
                    return game.move_attack(query, candidate_attacker, candidate_target, min(3, game.state.territories[candidate_attacker].troops - 1))


    if len(game.state.recording) < 4000:
        # We will check if anyone attacked us in the last round.
        new_records = game.state.recording[game.state.new_records:]
        enemy = None
        for record in new_records:
            match record:
                case MoveAttack() as r:
                    if r.defending_territory in set(my_territories):
                        enemy = r.move_by_player

        # If we don't have an enemy yet, or we feel angry, this player will become our enemy.
        if enemy != None:
            if bot_state.enemy == None or random.random() < 0.05:
                bot_state.enemy = enemy
        
        # If we have no enemy, we will pick the player with the weakest territory bordering us, and make them our enemy.
        else:
            weakest_territory = min(bordering_territories, key=lambda x: game.state.territories[x].troops)
            bot_state.enemy = game.state.territories[weakest_territory].occupier
            
        # We will attack their weakest territory that gives us a favourable battle if possible.
        enemy_territories = list(set(bordering_territories) & set(game.state.get_territories_owned_by(enemy)))
        move = attack_weakest(enemy_territories)
        if move != None:
            return move
        
        # Otherwise we will attack anyone most of the time.
        if random.random() < 0.8:
            move = attack_weakest(bordering_territories)
            if move != None:
                return move

    # In the late game, we will attack anyone adjacent to our strongest territories (hopefully our doomstack).
    else:
        strongest_territories = sorted(my_territories, key=lambda x: game.state.territories[x].troops, reverse=True)
        for territory in strongest_territories:
            move = attack_weakest(list(set(game.state.map.get_adjacent_to(territory)) - set(my_territories)))
            if move != None:
                return move

    return game.move_attack_pass(query)


def handle_troops_after_attack(game: Game, bot_state: BotState, query: QueryTroopsAfterAttack) -> MoveTroopsAfterAttack:
    """After conquering a territory in an attack, you must move troops to the new territory."""
    
    # First we need to get the record that describes the attack, and then the move that specifies
    # which territory was the attacking territory.
    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])

    # We will always move the maximum number of troops we can.
    return game.move_troops_after_attack(query, game.state.territories[move_attack.attacking_territory].troops - 1)


def handle_defend(game: Game, bot_state: BotState, query: QueryDefend) -> MoveDefend:
    """If you are being attacked by another player, you must choose how many troops to defend with."""

    # We will always defend with the most troops that we can.

    # First we need to get the record that describes the attack we are defending against.
    move_attack = cast(MoveAttack, game.state.recording[query.move_attack_id])
    defending_territory = move_attack.defending_territory
    
    # We can only defend with up to 2 troops, and no more than we have stationed on the defending
    # territory.
    defending_troops = min(game.state.territories[defending_territory].troops, 2)
    return game.move_defend(query, defending_troops)


def handle_fortify(game: Game, bot_state: BotState, query: QueryFortify) -> Union[MoveFortify, MoveFortifyPass]:
    """At the end of your turn, after you have finished attacking, you may move a number of troops between
    any two of your territories (they must be adjacent)."""


    # We will always fortify towards the most powerful player (player with most troops on the map) to defend against them.
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    total_troops_per_player = {}
    for player in game.state.players.values():
        total_troops_per_player[player.player_id] = sum([game.state.territories[x].troops for x in game.state.get_territories_owned_by(player.player_id)])

    most_powerful_player = max(total_troops_per_player.items(), key=lambda x: x[1])[0]

    # If we are the most powerful, we will pass.
    if most_powerful_player == game.state.me.player_id:
        return game.move_fortify_pass(query)
    
    # Otherwise we will find the shortest path between our territory with the most troops
    # and any of the most powerful player's territories and fortify along that path.
    candidate_territories = game.state.get_all_border_territories(my_territories)
    most_troops_territory = max(candidate_territories, key=lambda x: game.state.territories[x].troops)

    # To find the shortest path, we will use a custom function.
    shortest_path = find_shortest_path_from_vertex_to_set(game, most_troops_territory, set(game.state.get_territories_owned_by(most_powerful_player)))
    # We will move our troops along this path (we can only move one step, and we have to leave one troop behind).
    # We have to check that we can move any troops though, if we can't then we will pass our turn.
    if len(shortest_path) > 0 and game.state.territories[most_troops_territory].troops > 1:
        return game.move_fortify(query, shortest_path[0], shortest_path[1], game.state.territories[most_troops_territory].troops - 1)
    else:
        return game.move_fortify_pass(query)


def find_shortest_path_from_vertex_to_set(game: Game, source: int, target_set: set[int]) -> list[int]:
    """Used in move_fortify()."""

    # We perform a BFS search from our source vertex, stopping at the first member of the target_set we find.
    queue = deque()
    queue.appendleft(source)

    current = queue.pop()
    parent = {}
    seen = {current: True}

    while len(queue) != 0:
        if current in target_set:
            break

        for neighbour in game.state.map.get_adjacent_to(current):
            if neighbour not in seen:
                seen[neighbour] = True
                parent[neighbour] = current
                queue.appendleft(neighbour)

        current = queue.pop()

    path = []
    while current in parent:
        path.append(current)
        current = parent[current]

    return path[::-1]

if __name__ == "__main__":
    main()