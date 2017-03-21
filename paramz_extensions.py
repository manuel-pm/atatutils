from paramz.optimization import Optimizer


class opt_bfgs_simplex(Optimizer):
    def __init__(self, *args, **kwargs):
        Optimizer.__init__(self, *args, **kwargs)
        self.opt_name = "Mixed BFGS + Nelder-Mead simplex routine (via Scipy)"

    def opt(self, x_init, f_fp=None, f=None, fp=None):
        """
        The simplex optimizer does not require gradients.
        BFGS requires gradients
        """

        statuses = ['Converged', 'Maximum number of function evaluations made', 'Maximum number of iterations reached',
                    'Gradient and/or function calls not changing']

        opt_dict = {}
        if self.xtol is not None:
            opt_dict['xtol'] = self.xtol
        if self.ftol is not None:
            opt_dict['ftol'] = self.ftol
        if self.gtol is not None:
            print("WARNING: simplex doesn't have an gtol arg, so I'm going to ignore it")

        print(f_fp, f, fp)
        baboom

        #opt_result = optimize.fmin_bfgs(f, x_init, fp, disp=self.messages,
        #                                    maxiter=self.max_iters, full_output=True, **opt_dict)
        self.x_opt = opt_result[0]
        self.f_opt = f_fp(self.x_opt)[0]
        self.funct_eval = opt_result[4]
        self.status = rcstrings[opt_result[6]]

        #opt_result = optimize.fmin(f, x_init, (), disp=self.messages,
        #           maxfun=self.max_f_eval, full_output=True, **opt_dict)

        self.x_opt = opt_result[0]
        self.f_opt = opt_result[1]
        self.funct_eval = opt_result[3]
        self.status = statuses[opt_result[4]]
        self.trace = None

"""
        opt_dict = {}
        if self.xtol is not None:
            print("WARNING: bfgs doesn't have an xtol arg, so I'm going to ignore it")
        if self.ftol is not None:
            print("WARNING: bfgs doesn't have an ftol arg, so I'm going to ignore it")
        if self.gtol is not None:
            opt_dict['gtol'] = self.gtol

        opt_result = optimize.fmin_bfgs(f, x_init, fp, disp=self.messages,
                                            maxiter=self.max_iters, full_output=True, **opt_dict)
        self.x_opt = opt_result[0]
        self.f_opt = f_fp(self.x_opt)[0]
        self.funct_eval = opt_result[4]
        self.status = rcstrings[opt_result[6]]
"""
"""
class opt_SCG(Optimizer):
    def __init__(self, *args, **kwargs):
        if 'max_f_eval' in kwargs:
            warn("max_f_eval deprecated for SCG optimizer: use max_iters instead!\nIgnoring max_f_eval!", FutureWarning)
        Optimizer.__init__(self, *args, **kwargs)

        self.opt_name = "Scaled Conjugate Gradients"

    def opt(self, x_init, f_fp=None, f=None, fp=None):
        assert not f is None
        assert not fp is None

        opt_result = SCG(f, fp, x_init,
                         maxiters=self.max_iters,
                         max_f_eval=self.max_f_eval,
                         xtol=self.xtol, ftol=self.ftol,
                         gtol=self.gtol)

        self.x_opt = opt_result[0]
        self.trace = opt_result[1]
        self.f_opt = self.trace[-1]
        self.funct_eval = opt_result[2]
        self.status = opt_result[3]
"""
