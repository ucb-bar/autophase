import pickle
import numpy as np
def process_csv(filename='train_rand.pkl'):
  entries = []
  with open('random.log') as f:
    lines = f.readlines()

    for line in lines:
      data = line.split('],')
      obs = data[0].replace('[','').split(',')
      obs = list(map(lambda x: int(x), obs))

      tmp = data[1].split(',')
      act = int(tmp[0])
      rew = int(tmp[1])

      entries.append((obs, act, rew))

  output = open(filename, 'wb')
  pickle.dump(entries, output)
  output.close()

def load_data(filename='train_rand.pkl'):
  num_passes = 45
  pkl_file = open(filename, 'rb')
  entries = pickle.load(pkl_file)
  train_data = {}
  y_data = {}
  given_passes = []
  for p in range(num_passes):
      train_data[p]=[]
      y_data[p]=[]
  for (obs,act,rew) in entries:
    if(int(act) not in given_passes):
        given_passes.append(int(act))
    train_data[int(act)].append(obs)
    good_rew = 0
    if int(rew) > 0:
        good_rew = 1
    y_data[int(act)].append(good_rew)
  #obs, act, rew = entries[0]
  pkl_file.close()
  print("num passes found in pkl file",len(sorted(given_passes)))
  for p in range(num_passes):
      train_data[p] = np.array(train_data[p])
      y_data[p] = np.array(y_data[p])
  #print("{} {} {}".format(obs, act, rew))
  return train_data,y_data

#process_csv()


