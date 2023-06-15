    def plot_slow_roll_param_against_tau(self):
        super().plot_slow_roll_param_against_tau()

        y = -np.sqrt(16*np.pi/3)*self.ψ
        N = 3/4*np.exp(-y)
        ϵ = 3/(4*N**2)
        η = -1/N

        plt.semilogy(self.τ, η, "--", label=r"$\eta \approx -\frac{1}{N}$")
        plt.semilogy(self.τ, ϵ, "--", label=r"$\epsilon \approx \frac{3}{4N^2}$")
        plt.legend()

        plot_inset(plt.gca(), [0.21,0.59,0.35,0.35], xlim=[2695,2750], ylim=[1e2,1e10])
        plot_inset(plt.gca(), [0.21,0.14,0.35,0.35], xlim=[2695,2750], ylim=[0.9,2.1])

        plt.savefig(f"{self.subclassName}_slowroll-tau")
		
	\bibitem{BahngSchwarzild}
		Bahng, J. \& Schwarzschild, M. 1961.
		\textit{The Temperature Fluctuations in the Solar Granulation}.
		 \textit{\apj}, \textbf{134}, 337-342. \\
		\url{https://doi.org/10.1086/147163}
