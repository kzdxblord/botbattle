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
from risk_shared.records.record_player_eliminated import PublicRecordPlayerEliminated
from risk_shared.records.types.move_type import MoveType
import time

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
                
def get_territory_continent(territory_id: int) -> str:
    """Get the continent of a territory."""
    continent_map = {
        'NA': range(0, 9),
        'EU': range(9, 16),
        'AS': range(16, 28),
        'AF': range(32, 38),
        'AU': range(38, 42),
        'SA': range(28, 32)
    }
    for continent, territory_range in continent_map.items():
        if territory_id in territory_range:
            return continent
    raise ValueError(f"Invalid territory ID: {territory_id}")

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
    strategic_values = {territory: get_threat_territory(game, bot_state, territory) for territory in border_territories}

    # Calculate priority scores for each border territory
    priority_scores = {}
    for territory in border_territories:
        troops = game.state.territories[territory].troops
        threat = strategic_values[territory]
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
    adjacent_strength = 0 #40
    adjacent_territories = game.state.get_all_adjacent_territories([territory]) #
    continent_importance = continent_bonuses[territory_to_cont[territory]] #5
    strategic_position = 0 #5

    for i in adjacent_territories:
        if i not in my_territories:
            strength = game.state.territories[i].troops
            adjacent_strength += strength
    
    if territory in chokepoints:
        strategic_position = 5

    army_strength = scale(army_strength)
    adjacent_strength = scale(adjacent_strength, input_min=0, input_max=100, output_min=0, output_max=40)
    continent_importance = scale(continent_importance, input_min=0, input_max=7, output_min=0, output_max=5)

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

    # Calculate strategic values for all border territories
    strategic_values = calculate_strategic_values(game, bot_state, my_territories)

    # Sort territories by strategic value in descending order
    sorted_territories = sorted(strategic_values.items(), key=lambda x: x[1], reverse=True)

    # Allocate troops based on strategic value
    while total_troops > 0:
        for territory, _ in sorted_territories:
            if total_troops == 0:
                break
            
            troops_to_place = min(max(1, total_troops // 3), total_troops)

            if territory in border_territories:
                troops_to_place = min(troops_to_place + 1, total_troops)

            distributions[territory] += troops_to_place
            total_troops -= troops_to_place
   

    return game.move_distribute_troops(query, distributions)

def calculate_strategic_values(game: Game, bot_state: BotState, my_territories: list[int]) -> dict[int, float]:
    """Calculate strategic values for territories."""
    strategic_values = {}
    for territory in my_territories:
        # Base strategic value on number of adjacent enemy territories
        adjacent_territories = game.state.map.get_adjacent_to(territory)
        enemy_adjacent = sum(1 for adj in adjacent_territories if game.state.territories[adj].occupier != game.state.me.player_id)
        
        # Check if it's a chokepoint (has only one friendly adjacent territory)
        is_chokepoint = sum(1 for adj in adjacent_territories if adj in my_territories) == 1
        
        # Check if it's part of a continent we're close to controlling
        continent = get_territory_continent(territory)
        continent_control = calculate_continent_control(game, game.state.me.player_id)[continent]
        
        # Consider the threat level
        threat_level = get_threat_territory(game, bot_state, territory)
        
        # Calculate the strategic value
        strategic_value = (
            enemy_adjacent * 0.3 +  # More adjacent enemies increase value
            (1.5 if is_chokepoint else 0) +  # Bonus for chokepoints
            (continent_control * 2) +  # Bonus for territories in continents we're close to controlling
            (threat_level * 0.5)  # Consider threat level, but with less weight
        )
        
        strategic_values[territory] = strategic_value

    return strategic_values

# Define continents at the module level
CONTINENTS = {
    'NA': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'SA': [28, 29, 30, 31],
    'EU': [9, 10, 11, 12, 13, 14, 15],
    'AS': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    'AU': [38, 39, 40, 41],
    'AF': [32, 33, 34, 35, 36, 37]
}

def calculate_continent_control(game: Game, player_id: int) -> dict:
    """Calculate the percentage of each continent controlled by the player."""
    continent_control = {}
    player_territories = set(game.state.get_territories_owned_by(player_id))
    for continent, territories in CONTINENTS.items():
        control = len(set(territories) & player_territories) / len(territories)
        continent_control[continent] = control
    return continent_control

def evaluate_attack_opportunity(game: Game, attacker: int, defender: int) -> float:
    """Evaluate the opportunity of an attack based on various factors."""
    attacker_troops = game.state.territories[attacker].troops
    defender_troops = game.state.territories[defender].troops
    
    # Basic troop advantage
    troop_advantage = (attacker_troops - defender_troops) / max(defender_troops, 1)
    
    # Continental control factor
    attacker_continents = calculate_continent_control(game, game.state.territories[attacker].occupier) # type: ignore
    defender_continents = calculate_continent_control(game, game.state.territories[defender].occupier) # type: ignore
    
    # Get the continent of the defending territory
    defending_continent = get_territory_continent(defender)
    
    # Calculate the difference in control for the relevant continent
    continent_factor = attacker_continents[defending_continent] - defender_continents[defending_continent]
    
    # Strategic value of the defending territory
    strategic_value = len(game.state.map.get_adjacent_to(defender)) / 6  # Normalize by max possible connections
    
    # Bonus for completing a continent
    continent_completion_bonus = 0
    territories_in_continent = len(CONTINENTS[defending_continent])
    if defender_continents[defending_continent] == (territories_in_continent - 1) / territories_in_continent:
        continent_completion_bonus = 0.5  # Significant bonus for potentially completing a continent
    
    # Combine factors (weights can be adjusted)
    opportunity_score = (
        0.4 * troop_advantage +
        0.3 * continent_factor +
        0.2 * strategic_value +
        0.1 * continent_completion_bonus
    )
    
    return opportunity_score

def find_best_attack(game: Game) -> Union[tuple[int, int], None]:
    """Find the best attack opportunity."""
    my_territories = game.state.get_territories_owned_by(game.state.me.player_id)
    best_score = float('-inf')
    best_attack = None
    
    for attacker in my_territories:
        if game.state.territories[attacker].troops > 1:
            adjacent_enemies = set(game.state.map.get_adjacent_to(attacker)) - set(my_territories)
            for defender in adjacent_enemies:
                score = evaluate_attack_opportunity(game, attacker, defender)
                if score > best_score:
                    best_score = score
                    best_attack = (attacker, defender)
    
    return best_attack if best_score > 0 else None

def calculate_player_strength(game: Game, player_id: int) -> int:
    """Calculate the total strength of a player's army."""
    return sum(game.state.territories[t].troops for t in game.state.get_territories_owned_by(player_id))

def calculate_border_strength(game: Game, player_id: int) -> float:
    """Calculate the average strength of a player's border territories."""
    border_territories = game.state.get_all_border_territories(game.state.get_territories_owned_by(player_id))
    if not border_territories:
        return 0
    return sum(game.state.territories[t].troops for t in border_territories) / len(border_territories)

def get_strongest_enemy(game: Game) -> int:
    """Identify the strongest enemy player."""
    player_strengths = {
        player_id: calculate_player_strength(game, player_id)
        for player_id in game.state.players if player_id != game.state.me.player_id
    }
    return max(player_strengths, key=player_strengths.get) # type: ignore

def has_valid_attacks(game: Game) -> bool:
    """Check if there are any valid attacks available."""
    my_border_territories = game.state.get_all_border_territories(game.state.get_territories_owned_by(game.state.me.player_id))
    
    for territory in my_border_territories:
        if game.state.territories[territory].troops > 1:
            adjacent_territories = game.state.map.get_adjacent_to(territory)
            for adjacent in adjacent_territories:
                if game.state.territories[adjacent].occupier != game.state.me.player_id:
                    return True
    return False

def should_continue_attacking(game: Game) -> bool:
    """Decide whether to continue attacking, with a more aggressive approach."""
    if not has_valid_attacks(game):
        return False

    my_strength = calculate_player_strength(game, game.state.me.player_id)
    strongest_enemy_id = get_strongest_enemy(game)
    enemy_strength = calculate_player_strength(game, strongest_enemy_id)

    my_territories = len(game.state.get_territories_owned_by(game.state.me.player_id))
    total_territories = len(game.state.territories)
    control_percentage = my_territories / total_territories

    # Increased base probability for more aggression
    attack_probability = 0.5 + (0.15 * (my_strength / max(enemy_strength, 1) - 1))
    
    # Adjust based on game phase, encouraging more attacks in early and mid-game
    if control_percentage < 0.3:  # Early game
        attack_probability += 0.2
    elif control_percentage < 0.6:  # Mid game
        attack_probability += 0.1
    
    return random.random() < max(0, min(1, attack_probability))

def quick_evaluate_attack(game: Game, attacker: int, defender: int) -> float:
    """Quickly evaluate the opportunity of an attack, favoring aggression."""
    attacker_troops = game.state.territories[attacker].troops
    defender_troops = game.state.territories[defender].troops
    
    # More aggressive troop advantage calculation
    troop_advantage = (attacker_troops - defender_troops) / max(defender_troops, 1)
    
    # Quick strategic value calculation
    strategic_value = len(game.state.map.get_adjacent_to(defender)) / 6
    
    # Bonus for attacking weaker territories
    weakness_bonus = 1 / max(defender_troops, 1)
    
    # Combine factors with weights favoring aggression
    return troop_advantage * 0.5 + strategic_value * 0.3 + weakness_bonus * 0.2

def handle_attack(game: Game, bot_state: BotState, query: QueryAttack) -> Union[MoveAttack, MoveAttackPass]:
    """Improved attack strategy with more aggressive approach."""
    start_time = time.time()
    max_time = 0.5
    max_attempts = 100  # Increased from 10 to allow more attack attempts

    for _ in range(max_attempts):
        if time.time() - start_time > max_time:
            return game.move_attack_pass(query)

        if not should_continue_attacking(game):
            return game.move_attack_pass(query)

        best_attack = find_best_attack(game)
        if best_attack:
            attacker, defender = best_attack
            # More aggressive: always attack with maximum troops
            attack_troops = min(3, game.state.territories[attacker].troops - 1)
            return game.move_attack(query, attacker, defender, attack_troops)
    
    return game.move_attack_pass(query)


def calculate_frontier_pressure(game: Game, territory_id: int) -> float:
    """Calculate the pressure on a territory from enemy troops in adjacent territories."""
    adjacent_territories = game.state.map.get_adjacent_to(territory_id)
    enemy_troops = sum(
        game.state.territories[adj].troops
        for adj in adjacent_territories
        if game.state.territories[adj].occupier != game.state.me.player_id
    )
    return enemy_troops / len(adjacent_territories) if adjacent_territories else 0

def continent_completion_factor(game: Game, territory_id: int) -> float:
    """Calculate how close we are to completing the continent of the given territory."""
    continent = get_territory_continent(territory_id)
    continent_territories = [t for t in game.state.territories if get_territory_continent(t) == continent]
    owned_territories = [t for t in continent_territories if game.state.territories[t].occupier == game.state.me.player_id]
    return len(owned_territories) / len(continent_territories)

def handle_troops_after_attack(game: Game, bot_state: BotState, query: QueryTroopsAfterAttack) -> MoveTroopsAfterAttack:
    """Decide how many troops to move after a successful attack."""
    
    record_attack = cast(RecordAttack, game.state.recording[query.record_attack_id])
    move_attack = cast(MoveAttack, game.state.recording[record_attack.move_attack_id])
    
    attacking_territory = move_attack.attacking_territory
    conquered_territory = move_attack.defending_territory
    available_troops = game.state.territories[attacking_territory].troops - 1

    # Calculate strategic factors
    frontier_pressure = calculate_frontier_pressure(game, conquered_territory)
    completion_factor = continent_completion_factor(game, conquered_territory)
    
    # Base troops to move
    base_troops = max(3, int(available_troops * 0.8))  # Move at least 3 troops or 60% of available

    # Adjust based on frontier pressure
    pressure_adjustment = int(frontier_pressure * 1.5)  # Increase troops if there's high pressure
    
    # Adjust based on continent completion
    completion_adjustment = int(completion_factor * available_troops * 0.3)  # Move more if close to completing continent
    
    # Calculate final number of troops to move
    troops_to_move = min(
        available_troops,
        base_troops + pressure_adjustment + completion_adjustment
    )

    troops_to_move = troops_to_move if available_troops - troops_to_move > 0 else available_troops
    
    return game.move_troops_after_attack(query, troops_to_move)


def handle_defend(game: Game, bot_state: BotState, query: QueryDefend) -> MoveDefend:
    """Decide how many troops to defend with based on various strategic factors."""
    
    move_attack = cast(MoveAttack, game.state.recording[query.move_attack_id])
    attacking_territory = move_attack.attacking_territory
    defending_territory = move_attack.defending_territory
    attacking_troops = move_attack.attacking_troops
    
    # Get the number of troops we have on the defending territory
    our_troops = game.state.territories[defending_territory].troops
    
    # Calculate the importance of the defending territory
    territory_importance = calculate_territory_importance(game, defending_territory)
    
    # Calculate the risk of losing the territory
    loss_risk = calculate_loss_risk(attacking_troops, our_troops)
    
    # Determine the optimal number of troops to defend with
    if our_troops == 1:
        # We have no choice but to defend with 1 troop
        troops_to_defend = 1
    elif our_troops == 2:
        # Decide whether to defend with 1 or 2 troops based on territory importance and loss risk
        troops_to_defend = 2 if (territory_importance > 0.5 or loss_risk > 0.7) else 1
    else:
        # We have the option to defend with up to 2 troops
        if territory_importance > 0.8 or loss_risk > 0.9:
            troops_to_defend = 2
        elif territory_importance > 0.5 or loss_risk > 0.6:
            troops_to_defend = 2 if random.random() < 0.7 else 1  # 70% chance to defend with 2
        else:
            troops_to_defend = 1
    
    return game.move_defend(query, troops_to_defend)

def calculate_territory_importance(game: Game, territory_id: int) -> float:
    """Calculate the strategic importance of a territory."""
    continent = get_territory_continent(territory_id)
    continent_control = calculate_continent_control(game, game.state.me.player_id)[continent]
    
    # Check if it's a chokepoint
    adjacent_territories = game.state.map.get_adjacent_to(territory_id)
    friendly_adjacent = sum(1 for adj in adjacent_territories if game.state.territories[adj].occupier == game.state.me.player_id)
    is_chokepoint = friendly_adjacent == 1
    
    # Calculate importance based on various factors
    importance = (
        0.3 * continent_control +  # Higher importance if we control more of the continent
        0.3 * (1 - continent_control) +  # Also important if we're trying to gain control
        0.2 * (len(adjacent_territories) / 6) +  # More connections = more important
        0.2 * (2 if is_chokepoint else 1)  # Bonus for chokepoints
    )
    
    return min(1.0, importance)  # Ensure the value is between 0 and 1

def calculate_loss_risk(attacking_troops: int, defending_troops: int) -> float:
    """Calculate the risk of losing the territory based on troop numbers."""
    # These probabilities are approximate and could be fine-tuned with more accurate calculations
    if defending_troops == 1:
        return 0.7 if attacking_troops >= 2 else 0.4
    elif defending_troops == 2:
        if attacking_troops == 1:
            return 0.25
        elif attacking_troops == 2:
            return 0.45
        else:
            return 0.65
    else:
        return 0.0  # This shouldn't happen in Risk, but just in case


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