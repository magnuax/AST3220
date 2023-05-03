
        """
        Y_p, Y_n, Y_D, Y_T, Y_He3, Y_He4, Y_Li7, Y_Be7  = self.sol.y
        AY_sum = (1*Y_p   + 1*Y_n   + 2*Y_D   + 3*Y_T +
                 3*Y_He3 + 4*Y_He4 + 7*Y_Li7 + 7*Y_Be7)

        ax.semilogx(T, Y_n, label=r"$n$")
        ax.semilogx(T, Y_p, label=r"$p$")
        ax.semilogx(T, Y_D, label=r"$D$")
        ax.semilogx(T, Y_T, label=r"$T$")
        ax.semilogx(T, Y_He3, label=r"$He^3$")
        ax.semilogx(T, Y_He4, label=r"$He^4$")
        ax.semilogx(T, Y_Li7, label=r"$Li^7$")
        ax.semilogx(T, Y_Be7, label=r"$Be^7$")
        """


        """
        Y_p   = number_densities[0,:]
        Y_n   = number_densities[1,:]
        Y_D   = number_densities[2,:]
        Y_T   = number_densities[3,:]
        Y_He3 = number_densities[4,:]
        Y_He4 = number_densities[4,:]
        Y_Li7 = number_densities[6,:]
        Y_Be7 = number_densities[7,:]

        #Y_p,   Y_n,   Y_D,   Y_T, Y_He3, Y_He4, Y_Li7, Y_Be7

        """
            #mass_fractions = [self.sol.t, *self.sol.y]
            #np.save("mass_fractions.npy", mass_fractions)


    def import_solution(self, filename="mass_fractions.npy"):
        self.sol = np.load(filename)
        self.log_T = self.sol[0]
        self.Y     = self.sol[1:]

    N_eff = Y_relic[0,:]
    Y_p   = Y_relic[1,:]
    Y_n   = Y_relic[2,:]
    Y_D   = Y_relic[3,:]
    Y_T   = Y_relic[4,:]
    Y_He3 = Y_relic[5,:]
    Y_He4 = Y_relic[6,:]
    Y_Li7 = Y_relic[7,:]
    Y_Be7 = Y_relic[8,:]

    Y_Li7 = Y_Li7 + Y_Be7
    Y_He3 = Y_He3 + Y_T


    logY_D_p   = interp1d(N_eff, np.log(Y_D/Y_p)  , kind="cubic")
    logY_Li7_p = interp1d(N_eff, np.log(Y_Li7/Y_p), kind="cubic")
    logY_4xHe4 = interp1d(N_eff, np.log(4*Y_He4)  , kind="cubic")
    logY_He3   = interp1d(N_eff, np.log(Y_He3/Y_p), kind="cubic")

    N = 1000
    N_eff = np.linspace(1, 5, N)

    Y_D_p   = np.exp( logY_D_p(N_eff)   )
    Y_Li7_p = np.exp( logY_Li7_p(N_eff) )
    Y_4xHe4 = np.exp( logY_4xHe4(N_eff) )
    Y_He3   = np.exp( logY_He3(N_eff)   )


            Ω_b0  = Y_relic[0,:]
            Y_p   = Y_relic[1,:]
            Y_n   = Y_relic[2,:]
            Y_D   = Y_relic[3,:]
            Y_T   = Y_relic[4,:]
            Y_He3 = Y_relic[5,:]
            Y_He4 = Y_relic[6,:]
            Y_Li7 = Y_relic[7,:]
            Y_Be7 = Y_relic[8,:]

            Y_Li7 = Y_Li7 + Y_Be7
            Y_He3 = Y_He3 + Y_T

            logY_D_p   = interp1d(Ω_b0, np.log(Y_D/Y_p)  , kind="cubic")
            logY_Li7_p = interp1d(Ω_b0, np.log(Y_Li7/Y_p), kind="cubic")
            logY_4xHe4 = interp1d(Ω_b0, np.log(4*Y_He4)  , kind="cubic")
            logY_He3   = interp1d(Ω_b0, np.log(Y_He3/Y_p), kind="cubic")

            N = 1000
            Ω_b0 = np.linspace(0.01, 1, N)

            Y_D_p   = np.exp( logY_D_p(Ω_b0)   )
            Y_Li7_p = np.exp( logY_Li7_p(Ω_b0) )
            Y_4xHe4 = np.exp( logY_4xHe4(Ω_b0) )
            Y_He3   = np.exp( logY_He3(Ω_b0)   )


            
def plot_mass_fractions(self, filename):
    """
    Plots relative densities of neutron/proton against temperature
    """
    sol = np.load(filename)
    T   = np.exp(sol[0])
    # Create arrays with rel. densities, atomic number and element names
    Y_i = sol[1:]
    A_i = np.array([1, 1, 2, 3, 3, 4, 7, 7])
    X_i = np.array([r"$p$"   , r"$n$"   , r"$D$"   , r"$T$",
                    r"$He^3$", r"$He^4$", r"$Li^7$", r"$Be^7$"])
    # Use numpy broadcasting to calculate mass fraction
    AiYi = A_i[:,None]*Y_i

    fig, ax = plt.subplots(figsize=(8,5), tight_layout=True)
    ax.set_yscale("log")

    ax.semilogx(T, np.sum(AiYi, axis=0), "--k", label=r"$\Sigma_i A_i Y_i$")

    """
    for A, Y, X in zip(A_i, Y_i, X_i):
        ax.semilogx(T, A*Y, label=X)
    """
    ax.semilogx(T, A_i[0]*Y_i[0], label=X_i[0])
    ax.semilogx(T, A_i[1]*Y_i[1], label=X_i[1])
    ax.semilogx(T, A_i[2]*Y_i[2], label=X_i[2])
    Y_p0, Y_n0, _, _, _, _, _, _ = self._get_initial_conditions(T)
    plt.semilogx(T, Y_p0, color="C0", linestyle=":")
    plt.semilogx(T, Y_n0, color="C1", linestyle=":")

    ax.set_ylim(1e-3, 2)
    ax.set_xlim(1e11, 1e8)
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"Mass fraction $A_i Y_i$")
    plt.legend()
    plt.show()
