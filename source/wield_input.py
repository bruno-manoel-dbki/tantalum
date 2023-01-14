import wield


def energy1d(boundarie: dict):
    a = 1.0
    sigma = 0.1
    eps = 0.25
    order = 32
    tolerance = 1E-16
    grain1_phi1 = boundarie["right_phi1"]
    grain1_Phi = boundarie["right_PHI"]
    grain1_phi2 = boundarie["right_phi2"]
    
    grain2_phi1 = boundarie["left_phi1"]
    grain2_Phi = boundarie["left_PHI"]
    grain2_phi2 = boundarie["left_phi2"]

    trace_angle = boundarie["trace_angle"]


    X = [0,   0,    0,    0,    0, a/2, -a/2,  a/2, -a/2, a/2,  a/2, -a/2, -a/2]
    Y = [0, a/2,  a/2, -a/2, -a/2,   0,    0,    0,    0, a/2, -a/2,  a/2, -a/2]
    Z = [0, a/2, -a/2,  a/2, -a/2, a/2,  a/2, -a/2, -a/2,   0,    0,    0,    0]

    C1 = wield.CrystalGD(order, a, a, a, sigma, X, Y, Z, 1, True)
    C2 = C1

    #Rground = wield.createMatrixFromBungeEulerAngles(4,0,2)
    #ground  = wield.Surface(C1,Rground,C1,Rground,eps,tolerance)




    R1 = wield.createMatrixFromBungeEulerAngles(grain1_phi1,grain1_Phi,grain1_phi2)
    R2 = wield.createMatrixFromBungeEulerAngles(grain2_phi1,grain2_Phi,grain2_phi2)


    theta = trace_angle


    #print(theta)
    Rtheta1 = wield.createMatrixFromXAngle(theta)
    Rtheta2 = wield.createMatrixFromXAngle(theta)
    energy  = wield.SurfaceGD(C1,Rtheta1 @ R1,C2,Rtheta2 @ R2,eps,tolerance)


    return energy

