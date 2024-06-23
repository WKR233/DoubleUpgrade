import numpy as np
from collections import Counter
from itertools import combinations

class move_generator():
    def __init__(self, level, major):
        self.card_scale = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K']
        self.suit_set = ['s', 'h', 'c', 'd']
        self.point_order = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']
        self.Major = ["jo", "Jo"]
        self.major = major
        self.level = level
        self.point_order.remove(self.level)
        if self.major != 'n':
            self.Major = [self.major+point for point in self.point_order if point != self.level] + \
                [suit + level for suit in self.suit_set if suit != self.major] + [self.major + self.level] + self.Major
        else:
            self.Major = [suit + self.level for suit in self.suit_set] + self.Major
        
        
    def gen_single(self, deck, tgt):
        '''
        deck: player's deck
        tgt: target cardset(list of cardnames)
        '''
        moves = []
        if tgt[0] in self.Major:
            moves = [[p] for p in deck if p in self.Major]
            if len(moves) == 0: # No Major in deck
                moves = [[p] for p in deck]
        else:
            moves = [[p] for p in deck if p[0] == tgt[0][0] and p not in self.Major]
            if len(moves) == 0:
                moves = [[p] for p in deck]
        
        return moves 
    
    
    def gen_pair(self, deck, tgt):
        '''
        deck: player's deck
        tgt: target cardset(list of cardnames)
        '''
        moves = []
        if tgt[0] in self.Major:
            sel_deck = [p for p in deck if p in self.Major]
        else:
            sel_deck = [p for p in deck if p[0] == tgt[0][0] and p not in self.Major]
        sel_count = Counter(sel_deck)
        for k,v in sel_count.items():
            if v == 2:
                moves.append([k, k])
        if len(moves) == 0:
            if len(sel_deck) >= 2: # Generating cardset with same suit
                for i in range(len(sel_deck)-1):
                    for j in range(len(sel_deck)-i-1):
                        moves.append([sel_deck[i], sel_deck[i+j+1]])
            elif len(sel_deck) == 1:
                move_uni = sel_deck
                for p in deck:
                    if p != move_uni[0]:
                        moves.append(move_uni+[p])
            else:
                deck_cnt = Counter(deck)
                for k, v in deck_cnt:
                    if v == 2:
                        moves.append([k, k])
                if len(moves) == 0:
                    for i in range(len(deck)-1):
                        for j in range(len(deck)-i-1):
                            moves.append([deck[i], deck[i+j+1]])        
        return moves
    
    def gen_tractor(self, deck, tgt):
        moves = []
        tractor_len = len(tgt)
        pair_cnt = tractor_len // 2
        if tgt[0] in self.Major:
            sel_deck = [p for p in deck if p in self.Major]
        else:
            sel_deck = [p for p in deck if p[0] == tgt[0][0] and p not in self.Major]
        
        sel_count = Counter(sel_deck)
        sel_pairs = [k for k, v in sel_count.items() if v == 2] # Is actually list of cardname
        if "jo" in sel_pairs and "Jo" in sel_pairs and tractor_len == 4:
            moves.append(["jo", "jo", "Jo", "Jo"])
        if tgt[0] in self.Major:
            if self.major != 'n':
                trac_pairs = [p for p in sel_pairs if p[0] == self.major and p[1] != self.level]
                trac_pairs.sort(key=lambda x: self.point_order.index(x[1]))
            else:
                trac_pairs = []
        else:
            trac_pairs = sel_pairs + []
            trac_pairs.sort(key=lambda x: self.point_order.index(x[1]))
            
        if len(sel_deck) < len(tgt): # attaching cards with other suits
            other_deck = [p for p in deck if p not in sel_deck]
            move_uni = sel_deck
            sup_sets = list(combinations(other_deck, len(tgt)-len(sel_deck)))
            for cardset in sup_sets:
                moves.append(move_uni+list(cardset))
        
        else:  # attaching cards with same suits
            if len(sel_pairs) < pair_cnt:
                move_uni = [p for k in sel_pairs for p in [k, k]]
                sup_singles = [p for p in sel_deck if p not in sel_pairs] # enough to make a cardset
                sup_sets = list(combinations(sup_singles, tractor_len - len(sel_pairs)*2))
                for cardset in sup_sets:
                    moves.append(move_uni + list(cardset))
            elif len(trac_pairs) < pair_cnt: # can be compensated with sel_pairs
                pair_sets = list(combinations(sel_pairs, tractor_len//2))
                for pairset in pair_sets:
                    moves.append([p for k in pairset for p in [k, k]])
            else:
                for i in range(len(trac_pairs)-pair_cnt+1): # Try to retrieve a tractor
                    if trac_pairs[i+pair_cnt-1] == self.point_order[self.point_order.index(trac_pairs[i][1])+pair_cnt-1]:
                        pair_set = [[self.point_order[k], self.point_order[k]] \
                            for k in (self.point_order.index(trac_pairs[i][1]), self.point_order.index(trac_pairs[i][1])+pair_cnt-1)]
                        moves.append([p for pair in pair_set for p in pair])
                if len(moves) == 0:
                    pairsets = list(combinations(sel_pairs, tractor_len//2))
                    moves = [[p for k in pairset for p in [k, k]] for pairset in pairsets]
                            
        return moves  
                        
    def gen_throw(self, deck, tgt):
        level = self.level
        major = self.major
        outpok = []
        tgt_count = Counter(tgt)
        pos = []
        tractor = []
        suit = ''
        for k, v in tgt_count.items():
            if v == 2:
                if k != 'jo' and k != 'Jo' and k[1] != level: # 大小王和级牌当然不会参与拖拉机
                    pos.append(self.point_order.index(k[1]))
                    suit = k[0]
        if len(pos) >= 2:
            pos.sort()
            tmp = []
            suc_flag = False
            for i in range(len(pos)-1):
                if pos[i+1]-pos[i] == 1:
                    if not suc_flag:
                        tmp = [suit + self.point_order[pos[i]], suit + self.point_order[pos[i]], suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]]
                        del tgt_count[suit + self.point_order[pos[i]]]
                        del tgt_count[suit + self.point_order[pos[i+1]]] # 已计入拖拉机的，从牌组中删去
                        suc_flag = True
                    else:
                        tmp.extend([suit + self.point_order[pos[i+1]], suit + self.point_order[pos[i+1]]])
                        del tgt_count[suit + self.point_order[pos[i+1]]]
                elif suc_flag:
                    tractor.append(tmp)
                    suc_flag = False
            if suc_flag:
                tractor.append(tmp)
        # 对牌型作基础的拆分 
        for k,v in tgt_count.items(): 
            outpok.append([k for i in range(v)])
        outpok.extend(tractor)
        
        moves = []
        move_uni = []
        self.reg_generator(deck, move_uni, outpok, moves)
        
        return moves


    def reg_generator(self, deck, move, outpok, moves):
        if len(outpok) == 0:
            moves.append(move)
            return
        tgt = outpok[-1] + []
        new_pok = outpok[:-1] + []
        if len(tgt) > 2:
            move_unis = self.gen_tractor(deck, tgt)
        elif len(tgt) == 2:
            move_unis = self.gen_pair(deck, tgt)
        elif len(tgt) == 1:
            move_unis = self.gen_single(deck, tgt)
        
        for move_uni in move_unis:
            new_move = move + move_uni
            new_deck = deck + []
            for p in move_uni:
                new_deck.remove(p)
            self.reg_generator(new_deck, new_move, new_pok, moves)
            
        return
            
    def gen_all(self, deck): # Generating all cardset options
        moves = []
        suit_decks = []
        major_deck = [p for p in deck if p in self.Major]
        for i in range(4):
            suit_decks.append([p for p in deck if p[0] == self.suit_set[i] and p not in self.Major])
        # Do the major first
        major_count = Counter(major_deck)
        # Adding in all pairs and singles
        for k, v in major_count.items():
            if v == 1:
                moves.append([k])
            if v == 2:
                moves.append([k, k])
        # Adding in tractors
        if "jo" in major_count and major_count["jo"] == 2 and "Jo" in major_count and major_count["Jo"] == 2:
            moves.append(["jo", "jo", "Jo", "Jo"])
        trac_major = [k for k,v in major_count.items() if v == 2 and k[1] != self.level and k[1] != 'o']
        trac_major.sort(key=lambda x: self.point_order.index(x[1]))
        tracstreak = []
        for i in range(len(trac_major)):
            if len(tracstreak) == 0 or self.point_order.index(trac_major[i][1]) - self.point_order.index(tracstreak[-1][1]) > 1: # begin a new tracstreak
                tracstreak = [trac_major[i], trac_major[i]]
            else:
                tracstreak.extend([trac_major[i], trac_major[i]])
                moves.append(tracstreak+[])
                
        for suit_deck in suit_decks:
            suit_count = Counter(suit_deck)
            # Adding in all pairs and singles
            for k, v in suit_count.items():
                if v == 1:
                    moves.append([k])
                if v == 2:
                    moves.append([k, k])
            # Adding in tractors
            tracstreak = []
            trac_suit = [k for k,v in suit_count.items() if v==2]
            trac_suit.sort(key=lambda x: self.point_order.index(x[1]))
            for i in range(len(trac_suit)):
                if len(tracstreak) == 0 or self.point_order.index(trac_suit[i][1]) - self.point_order.index(tracstreak[-1][1]) > 1: # begin a new tracstreak
                    tracstreak = [trac_suit[i], trac_suit[i]]
                else:
                    tracstreak.extend([trac_suit[i], trac_suit[i]])
                    moves.append(tracstreak+[])
                    
        return moves
        
        