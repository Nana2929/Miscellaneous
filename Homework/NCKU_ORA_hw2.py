#%%
# (n-1) sample variance; for population variance use pvariance
from statistics import variance, mean
import time
import gurobipy as gbp  # Python3.8.12
import numpy as np
from typing import Final, List, Tuple

#%%

D: Final[Tuple[int]] = (2.5, 3, 20)
RATES: Final[Tuple[int]] = (0.8, 1, 1.2)


def getY(N: int):
    """Get a set of y-rated yields for the 3 crops
    Args:
        N (int): sample size (to approximate a continuous distribution)
    """
    mu, sigma = 1, 0.1  # mean and standard deviation
    C = len(D)
    realized_Y = []
    for _ in range(N):
        yield_rate = np.random.normal(mu, sigma, 1000)
        curr_y = [D[j] * yield_rate[j] for j in range(C)]
        realized_Y.append(curr_y)
    return realized_Y


def solve(D: List[int]):
    # SOLVE under a certain scenario
    model = gbp.Model('agri')
    X = [0 for _ in range(3)]
    W = [0 for _ in range(4)]
    Y = [0 for _ in range(2)]
    for i in range(3):
        X[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'x{i+1}')
    for i in range(4):
        W[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'w{i+1}')
    for i in range(2):
        Y[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'y{i+1}')
    # === constraints ===
    model.addConstr(gbp.quicksum(X[i]
                    for i in range(3)) <= 500, name='total acreage')
    model.addConstr(D[0]*X[0] + Y[0] - W[0] >= 200, name='wheat')
    model.addConstr(D[1]*X[1] + Y[1] - W[1] >= 240, name='corn')
    model.addConstr(D[2]*X[2] - W[2] - W[3] >= 0, name='sugar beets')
    model.addConstr(
        W[2] <= 6000, name='sugar beets; yield tons below policy threshold')
    model.setObjective(
        170*W[0] - 238 * Y[0] - 150*X[0] +
        150*W[1] - 210 * Y[1] - 230*X[1] +
        36*W[2] + 10*W[3] - 260*X[2], gbp.GRB.MAXIMIZE)
    model.optimize()
    return model, X


# ====================== b ========================
#%%
# ======== EV =========
solved, Xev = solve(D)
print(f'EV Solution')
print(solved.objval)
Xev = [v.x for v in Xev]


# ======== WS =========
print(f'WS Solution')
WSobj = {}
for rate in RATES:
    scenarioD = [D[i] * rate for i in range(3)]
    model, _ = solve(scenarioD)
    WSobj[rate] = model.objval
    print('Objective value:', model.objVal)
    for v in model.getVars():
        print(f'{v.varName} = {v.x}')

# ====================== c ========================
# Defining model of RP problem


def trans(w):
    if w == 0:
        return '_low_'
    elif w == 1:
        return '_medium_'
    else:
        return '_high_'


def RPsolve(D: List[List[int]], N: int):
    # N: the number of scenarios
    model = gbp.Model('agri-rp')
    X = [0 for _ in range(3)]
    W = [[0 for _ in range(4)] for w in range(N)]
    Y = [[0 for _ in range(2)] for w in range(N)]
    for i in range(3):
        X[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'x{i+1}')
    for w in range(N):
        for i in range(4):
            W[w][i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'w{trans(w)}{i+1}')
        for i in range(2):
            Y[w][i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'y{trans(w)}{i+1}')
    model.addConstr(gbp.quicksum(X[i]
                    for i in range(3)) <= 500, name='total acreage')
    for w in range(N):
        model.addConstr(D[w][0]*X[0] + Y[w][0] - W[w][0]
                        >= 200, name=f'{trans(w)} wheat')
        model.addConstr(D[w][1]*X[1] + Y[w][1] - W[w][1]
                        >= 240, name=f'{trans(w)} corn')
        model.addConstr(D[w][2]*X[2] - W[w][2] - W[w][3] >=
                        0, name=f'{trans(w)} sugar beets')
        model.addConstr(
            W[w][2] <= 6000, name=f'{trans(w)}, sugar beets; yield tons below policy threshold')
    model.setObjective(-150*X[0] - 230*X[1] - 260*X[2] +
                       1/N * gbp.quicksum(170*W[w][0] - 238 * Y[w][0] + 150*W[w][1] - 210 * Y[w][1] + 36*W[w][2] + 10*W[w][3] for w in range(N)), gbp.GRB.MAXIMIZE)
    model.optimize()
    return model, W, Y


# ====================== d ========================
rpD = [[rate * x for x in D] for rate in RATES]
rpmodel, Wsto, Ysto = RPsolve(rpD, N=3)
print(f'RP Solution')
RPObj = rpmodel.objVal
for v in rpmodel.getVars():
    print(f'{v.varName} = {v.x}')

# ====================== e ========================
Wsto = [[Wsto[w][i].x for i in range(4)] for w in range(3)]
Ysto = [[Ysto[w][i].x for i in range(2)] for w in range(3)]


def getEEV(X: List[int], N: int,
           Wsto: List[List[int]],
           Ysto: List[List[int]]):
    # X: the mean EV solution
    # Wsto, Ysto: the optimal of stochastic part of the RP solution
    return -150*X[0] - 230*X[1] - 260*X[2] +\
        1/N * sum(170*Wsto[w][0] - 238 * Ysto[w][0] + 150*Wsto[w][1] - 210 *
                  Ysto[w][1] + 36*Wsto[w][2] + 10*Wsto[w][3] for w in range(N))


# For maximization problem:
EVPI = sum(WSobj.values())/len(WSobj) - RPObj
VSS = RPObj - getEEV(Xev, N=3, Wsto=Wsto, Ysto=Ysto)
print(f'EVPI: {EVPI}')
print(f'VSS: {VSS}')


# ======================= g ======================
def SPN(N: int, Ds: List[List[int]]):
    model = gbp.Model('agri-spn')
    X = [0 for _ in range(3)]
    for i in range(3):
        X[i] = model.addVar(lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'x{i+1}')
    model.addConstr(gbp.quicksum(X[i]
                                 for i in range(3)) <= 500, name='total acreage')
    w_objs = 0
    for wid, D in enumerate(Ds):
        # N 個 scenarios
        W = [0 for _ in range(4)]
        Y = [0 for _ in range(2)]
        for i in range(4):
            W[i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'w{wid},{i+1}')
        for i in range(2):
            Y[i] = model.addVar(
                lb=0, vtype=gbp.GRB.CONTINUOUS, name=f'y{wid},{i+1}')
        # === constraints ===
        model.addConstr(D[0]*X[0] + Y[0] - W[0] >= 200, name=f'{wid},wheat')
        model.addConstr(D[1]*X[1] + Y[1] - W[1] >= 240, name=f'{wid},corn')
        model.addConstr(D[2]*X[2] - W[2] - W[3] >=
                        0, name=f'{wid},sugar beets')
        model.addConstr(
            W[2] <= 6000, name=f'{wid}, sugar beets; yield tons below policy threshold')
        w_obj = 170*W[0] - 238 * Y[0] - 150*X[0] + 150*W[1] - \
            210 * Y[1] - 230*X[1] + 36*W[2] + 10*W[3] - 260*X[2]
        w_objs += w_obj
    model.setObjective(w_objs/N, gbp.GRB.MAXIMIZE)
    model.optimize()
    return model, X


# spnmodel = SPN(N=3, Ds=getY(N=30))


def XDgetObj(X, D):
    # wheat
    wheatProfit = - X[0] * 150
    if X[0] * D[0] <= 200:
        wheatProfit += (200 - (X[0] * D[0])) * (-238)
    else:
        wheatProfit += ((X[0] * D[0]) - 200) * 170
    # corn
    cornProfit = - X[1] * 230
    if X[1] * D[1] <= 240:
        cornProfit += (240 - (X[1] * D[1])) * (-210)
    else:
        cornProfit += ((X[1] * D[1]) - 240) * 150
    # sugar beets
    sugarBeetsProfit = - X[2] * 260
    if X[2] * D[2] <= 6000:
        sugarBeetsProfit += (X[2] * D[2]) * 36
    else:
        sugarBeetsProfit += ((X[2] * D[2]) - 6000) * 10 + 6000 * 36
    return wheatProfit + wheatProfit + sugarBeetsProfit


M = 15
LowerBoundDistr, UpperBoundDistr = [], []
LowerBoundX = []  # = UpperBoundX
for m in range(M):
    batchW = getY(N=30)
    spnmodel, spnX = SPN(N=3, Ds=batchW)
    # ===== LOWER BOUND ======
    LowerBoundDistr.append(spnmodel.objVal)
    spnX = [spnX[i].x for i in range(3)]
    LowerBoundX.append(spnX)
    # ===== UPPER BOUND ======
    UpperBoundCurr = 0
    for w in batchW:
        # each w is a realized yields of 3 crops
        UpperBoundCurr += XDgetObj(spnX, w)
    UpperBoundDistr.append(UpperBoundCurr/len(batchW))

Lnm = mean(LowerBoundDistr)
Slm = variance(LowerBoundDistr, xbar=Lnm)
# By table lookup, z\alpha/2 = 1.96 if \alpha = 0.05
# Calculate margin of error (half-width)
za2 = 1.96
MoE_l = za2 * (Slm/len(LowerBoundDistr))**0.5
print(f'Lower Bound CI:[{Lnm} - {MoE_l},{Lnm} + {MoE_l}]')

Unt = mean(UpperBoundDistr)
Su = variance(UpperBoundDistr, xbar=Unt)
MoE_u = za2 * (Su/len(UpperBoundDistr))**0.5
print(f'Upper Bound CI:[{Unt} - {MoE_u},{Unt} + {MoE_u}]')

# Lower Bound CI:[1133049.9786766106 - 15773.725592053115,1133049.9786766106 + 15773.725592053115]
# Upper Bound CI:[132383.22659538672 - 2832.4989481191024,132383.22659538672 + 2832.4989481191024]


# %%
