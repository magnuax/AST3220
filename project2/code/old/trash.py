
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
