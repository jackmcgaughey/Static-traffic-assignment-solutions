#method of successive averages to solve the traffic assignment problem

alpha, beta = 0.15, 4.0 
def BPR(t_init, alpha, beta, capacity, x):
  return t_init*(1 + alpha*(x / capacity)**beta)

def travel_times(x_init, link_data):
  times = []
  i = 0
  for t0, u0 in link_data:
    times.append(BPR(t0,0.15,4,u0,x_init[i]))
    i+=1
  return times

def relative_gap(x,t,demand):
  TSTT = 0
  j=0 
  for i in x:
    TSTT += i * t[j]
    j+=1
  SPTT = min(t) * demand
  return (TSTT / SPTT) - 1

def new_x(lam, x, x_star):
  new = []
  j=0
  for i in x:
    new.append((x_star[j] * lam) + (i * (1-lam)))
    j+=1
  return new

#data from simple network
#because each node has a different initial flow and capacity,
#each link needs to have a specifed t0 and u0 (capacity)
link_data = [(10.0,2.0),(20.0,4.0),(25.0,3.0)]
x = [0,0,0]
iterator = 1
demand = 10
epsilon = 1e-4
x = [0 for i in x]
t = travel_times(x,link_data)
x_hat = []
for i in range(len(x)):
  if i == t.index(min(t)):
    x_hat.append(demand)
  else:
    x_hat.append(0)
t = travel_times(x_hat,link_data)
x = x_hat

rg = relative_gap(x_hat,t,demand)

while rg > epsilon:
  lam = 1 / iterator
  #define x_star as all or nothing assignment
  x_star = []
  for i in range(len(x)):
    if i == t.index(min(t)):
      x_star.append(demand)
    else:
      x_star.append(0)
  x_hat = new_x(lam, x_hat, x_star)
  t = travel_times(x_hat,link_data)
  #shift flow onto shortest path heer
  rg = relative_gap(x_hat, t, demand)
  iterator +=1
from IPython import embed; embed(colors='neutral')
print(x_hat)
