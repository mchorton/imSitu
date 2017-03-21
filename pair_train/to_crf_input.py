import numpy as np
import sys
import torch

file_argvalues = open(sys.argv[1])
file_nouns = open(sys.argv[2])

generated_vectors = sys.argv[3]
role2Id = sys.argv[4]
noun2Id = sys.argv[5]

outfile = sys.argv[6]
outdir = sys.argv[7]

role2Id = torch.load(role2Id)
noun2Id = torch.load(noun2Id)


chimera_id_noun = {}
for (k,v) in noun2Id.items(): chimera_id_noun[v] = k

chimera_id_role = {}
for (k,v) in role2Id.items(): chimera_id_role[v] = k

noun_id = {}
id_noun = {}
for line in file_nouns.readlines():
  tabs = line.split("\t")
  _id = tabs[0]
  noun = tabs[1].strip()
  noun_id[noun] = _id
  noun_id[_id] = noun

role_verb = {}
role_index = {}
role_length = {}
role_map = {}
verb_nroles = {}
verb_verbid = {}

for line in file_argvalues.readlines():
  tabs = line.split("\t")
  verb = tabs[0]
  role = tabs[1].lower()
  verbid = tabs[2]
  index = tabs[3]
  length = tabs[4]  

  verb_verbid[verb] = verbid

  if verb not in verb_nroles: verb_nroles[verb] = 0
  verb_nroles[verb] += 1

  _id = (verb,role)
  role_verb[_id] = verb
  role_index[_id] = index
  role_length[_id] = length
  
  _map = {}
  for i in range(0,2*int(length)-1,2):
    global_id = tabs[i+5] 
    local_id = tabs[i+6]
    _map[global_id] = local_id.strip()
  
  role_map[_id] = _map

vector_data = torch.load(generated_vectors)
of = open(outfile, "w")
i = 0
for row in vector_data:
  if i % 1000 == 0 : print str(i) + "/" + str(len(vector_data)) 
  _input = row[0][0]
  _target = row[0][1]
  _output = row[1]
  
  role0 = chimera_id_role[_input[0]]
  verb = role_verb[role0]
  
  #establish the initial semantics
  l = verb_nroles[verb]
  semantics = {}
  for j in range(0, l):
    role = chimera_id_role[_input[2*j]]
    noun = chimera_id_noun[_input[2*j+1]]
    semantics[role] = noun

  target_role = chimera_id_role[_input[12]]
  source_noun = chimera_id_noun[_input[13]]
  target_noun = chimera_id_noun[_input[14]]
  
  semantics[target_role] = target_noun

  weight = _target[-1]
 
  #convert the sematics to imsitu format
  output = []
  for j in range(0,7): output.append(-1)
 
  output[0] = verb_verbid[verb] 
  for (k,v) in semantics.items():
    if v != "" : local_id = role_map[k][noun_id[v]]
    else: local_id = role_map[k]["-1"] 
    index = role_index[k]
    output[int(index)+1] = local_id

  filename = "chimera_{1}_{0}.jpg".format(verb,i)
  npvector = _output.numpy()
  npvector.astype(np.float32).tofile(outdir + "/" +filename)
  
  r = []
  for k in range(0,3): r.append("\t".join(map(str,output)))
  r.append(weight)

  crf_row = filename + "\t" + "\t".join(map(str,r))
  of.write(crf_row + "\n")
  i+=1

 
