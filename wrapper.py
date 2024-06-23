import numpy as np
import torch

class cardWrapper:
    def __init__(self, suit_sequence=['s', 'h', 'c', 'd'], point_sequence = ['2','3','4','5','6','7','8','9','0','J','Q','K','A']):
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.suit_sequence = suit_sequence
        self.point_sequence = point_sequence
        self.J_pos = self.suit_sequence.index('h')
        self.j_pos = self.suit_sequence.index('s')
        
    def name2pos(self, cardname):
        if cardname[0] == "J":
            return (self.J_pos, 13)
        if cardname[0] == "j":
            return (self.j_pos, 13)
        pos = (self.suit_sequence.index(cardname[0]), self.point_sequence.index(cardname[1]))
        return pos
    
    def pos2name(self, cardpos):
        if cardpos[1] == 13:
            if cardpos[0] == self.j_pos:
                return "jo"
            if cardpos[0] == self.J_pos:
                return "Jo"
            else:
                raise "Card not exists."
        
        return self.suit_sequence[cardpos[0]] + self.point_sequence[cardpos[1]]
    
    # adding cards to a cardset 
    def add_card(self, cardset: np.array, cards): 
    # cardset: np.array(2,4,14)
    # cards: list[str(2)], cardnames.
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[0, card_pos[0], card_pos[1]] == 0:
                cardset[0, card_pos[0], card_pos[1]] = 1
            elif cardset[1, card_pos[0], card_pos[1]] == 0:
                cardset[1, card_pos[0], card_pos[1]] = 1    
            else:
                raise "More than two cards with same suits and points. Please recheck."

        return cardset
    
    # removing cards from cardset
    def remove_card(self, cardset: np.array, cards):
        for card in cards:
            card_pos = self.name2pos(card)
            if cardset[1, card_pos[0], card_pos[1]] != 0:
                cardset[1, card_pos[0], card_pos[1]] = 0
            elif cardset[0, card_pos[0], card_pos[1]] != 0:
                cardset[0, card_pos[0], card_pos[1]] = 0
            else:
                raise "Card not in cardset! Please recheck."

        return cardset
    
    # From cardset to cardnames
    def Unwrap(self, cardset): 
        cards = []
        card_poses = np.nonzero(cardset)
        for i in range(card_poses[0].size):
            card_name = self.pos2name((card_poses[1][i], card_poses[2][i]))
            cards.append(card_name)
        
        return cards

    def obsWrap(self, obs, options):
        '''
        Wrapping the observation and craft the action_mask
        obs: raw obs from env
        '''
        id = obs['id']
        major_mat = np.zeros((2,4,14))
        deck_mat = np.zeros((2,4,14))
        hist_mat = np.zeros((8,4,14)) # Holding no more than 4 sets of cards
        played_mat = np.zeros((8,4,14))
        option_mat = np.zeros((108,4,14))
        
        self.add_card(major_mat, obs['major'])
        self.add_card(deck_mat, obs['deck'])
        for i in range(len(obs['history'])):
            self.add_card(hist_mat[i*2:(i+1)*2], obs['history'][i])
        played_cards = obs['played'][id:]+obs['played'][:id]
        for i in range(len(played_cards)):
            self.add_card(played_mat[i*2:(i+1)*2], played_cards[i])
        for i in range(len(options)):
            if i*2 >= option_mat.shape[0]:
                break
            self.add_card(option_mat[i*2:(i+1)*2], options[i])
        
        action_mask = np.zeros(54)
        action_mask[:len(options)] = 1
        
        return np.concatenate((major_mat, deck_mat, hist_mat, played_mat, option_mat)), action_mask
    
        