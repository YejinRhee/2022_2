import numpy as np
from w2v_utils import *

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """ 
    if np.all(u == v):
        return 1 
    dot = np.dot(u,v)  
    norm_u = np.linalg.norm(u) 
    norm_v = np.linalg.norm(v) 
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0
    cosine_similarity = dot/(norm_u*norm_v) 
    
    return cosine_similarity
 

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """ 
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
     
    e_a = word_to_vec_map.get(word_a)
    e_b = word_to_vec_map.get(word_b)
    e_c = word_to_vec_map.get(word_c) 
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output
     
    for w in words:    
        if w == word_c:
            continue
        cosine_sim = cosine_similarity(np.subtract(e_b,e_a), np.subtract(word_to_vec_map.get(w),e_c))
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w 
        
    return best_word


def complete_analogy_test(target):
    a = [3, 3] # Center at a
    a_nw = [2, 4] # North-West oriented vector from a
    a_s = [3, 2] # South oriented vector from a
    
    c = [-2, 1] # Center at c 
    word_to_vec_map = {'a': a,
                       'synonym_of_a': a,
                       'a_nw': a_nw, 
                       'a_s': a_s, 
                       'c': c, 
                       'c_n': [-2, 2], # N
                       'c_ne': [-1, 2], # NE
                       'c_e': [-1, 1], # E
                       'c_se': [-1, 0], # SE
                       'c_s': [-2, 0], # S
                       'c_sw': [-3, 0], # SW
                       'c_w': [-3, 1], # W
                       'c_nw': [-3, 2] # NW
                      } 
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])
            
    assert(target('a', 'a_nw', 'c', word_to_vec_map) == 'c_nw')
    assert(target('a', 'a_s', 'c', word_to_vec_map) == 'c_s')
    assert(target('a', 'synonym_of_a', 'c', word_to_vec_map) != 'c'), "Best word cannot be input query"
    assert(target('a', 'c', 'a', word_to_vec_map) == 'c')
 
    
complete_analogy_test(complete_analogy)
  
  
  
  
  
  
  
  
  
  
