Team: The Bots

New helper functions created: 


Get_territory_continent: returns the continent name based on the territory id

Scale: used for scaling weights when getting the threat value of territories (see helper function 3)

Get_threat_territory: assesses a territory’s threat based on number of troops in the territory, number of enemy troops in adjacent territories, whether the territory is in a chokepoint, and it’s continent importance. Returns a threat value from 0 to 1

Calculate_strategic_value: based on number of adjacent enemy territories, whether the territory is in a chokepoint or part of a continent that we are close to controlling, and threat level (func. 3)

Calculate_continent_control: Used in previous function. Returns a percentage indicating continental control

Evaluate_attack_opportunity: considers difference in strength of troops, continental control, and strategic value. Returns a float indicating attack opportunity

Find_best_attack: calls previous function for all territories. Best attack has highest attack opportunity value

Calculate_player_strength: gets the total number of troops 

Calculate_border_strength: gets the number of troops on borders

Get_strongest_enemy: returns player id of strongest player based on func. 8

Has_valid_attacks: returns true if player can continue legally attacking

Should_continue_attacking: considers player strength, enemy strength, and the strongest enemy to assess when to stop attacking

Quick_evaluate_attack: similar to func. 6, but uses a weakness factor (how weak the territory is in comparison to the attacking territory)  instead of continental control

Calculate_frontier_pressure: assesses how much enemy pressure is on a territory (how easy the territory is to be captured by enemies)

Calculate_loss_risk: determines probability of winning a battle (based on troops numbers)


Overview of strategy: 

Territory claim phase: Gets a list of available territories, and prioritizes getting territories that are both adjacent and would give a continent bonus. If this is not possible, a territory with the highest strategic value is chosen instead

Placing initial troops: The strategic value of bordering territories are calculated. Territories with the highest values are given troops

Redeeming cards: Redeems sets of cards in a game to maximize troops and strategic advantage, prioritizes cards from territories owned by the player, and continues redeeming as long as conditions permit, finally returning the list of redeemed card sets.

Distributing troops: Calculates threat level of bordering territories. Gives troops to 3 most threatened territories. Remaining troops are distributed to the remaining threatened territories

Attacking: Find the best attack based on strength of troops, opportunity to get a continent bonus, number of enemy troops, and threat level. Next best attack is calculated, and assesses whether it is worth continuing to attack. 

Distributing troops after an attack: Assesses how many troops to distribute based on  pressure from enemy territories, and how close maintaining the territory is to giving us a full continent

Defending: Determines how important keeping the territory is, and the probability of success based on the number of troops we possess in the territory. If it is not that important we defend with 1. Otherwise, we defend with 2. 

Fortifying: Untouched from complex_example

