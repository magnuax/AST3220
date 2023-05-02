
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
