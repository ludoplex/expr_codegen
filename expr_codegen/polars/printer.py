from sympy import Basic, Function, StrPrinter
from sympy.printing.precedence import precedence


# TODO: 如有新添加函数，需要在此补充对应的打印代码

class PolarsStrPrinter(StrPrinter):
    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # See if the class of expr is known, or if one of its super
            # classes is known, and use that print function
            # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # Function('gamma') does not get dispatched to _print_gamma
            classes = type(expr).__mro__
            # if AppliedUndef in classes:
            #     classes = classes[classes.index(AppliedUndef):]
            # if UndefinedFunction in classes:
            #     classes = classes[classes.index(UndefinedFunction):]
            # Another exception: if someone subclasses a known function, e.g.,
            # gamma, and changes the name, then ignore _print_gamma
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                                    c.__name__ == classes[0].__name__ or \
                                    c.__name__.endswith("Base")) + classes[i:]
            for cls in classes:
                printmethodname = f'_print_{cls.__name__}'
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def _print_Symbol(self, expr):
        # return expr.name
        return f"pl.col('{expr.name}')"

    def _print_Equality(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}=={self.parenthesize(expr.args[1], PREC)}"

    def _print_if_else(self, expr):
        return f"pl.when({self._print(expr.args[0])}).then({self._print(expr.args[1])}).otherwise({self._print(expr.args[2])})"

    def _print_ts_mean(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_mean({self._print(expr.args[1])})"

    def _print_ts_std_dev(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_std({self._print(expr.args[1])}, ddof=0)"

    def _print_ts_arg_max(self, expr):
        return f"_rolling_argmax({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_ts_arg_min(self, expr):
        return f"_rolling_argmin({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_ts_product(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_apply(lambda x: np.product(x.to_numpy()), window_size={self._print(expr.args[1])})"

    def _print_ts_max(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_max({self._print(expr.args[1])})"

    def _print_ts_min(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_min({self._print(expr.args[1])})"

    def _print_ts_delta(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.diff({self._print(expr.args[1])})"

    def _print_ts_delay(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.shift({self._print(expr.args[1])})"

    def _print_ts_corr(self, expr):
        return f"pl.rolling_corr({self._print(expr.args[0])}, {self._print(expr.args[1])}, window_size={self._print(expr.args[2])}, ddof=0)"

    def _print_ts_covariance(self, expr):
        return f"pl.rolling_cov({self._print(expr.args[0])}, {self._print(expr.args[1])}, window_size={self._print(expr.args[2])}, ddof=0)"

    def _print_ts_rank(self, expr):
        return f"_rolling_rank({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_ts_sum(self, expr):
        PREC = precedence(expr)
        return f"{self.parenthesize(expr.args[0], PREC)}.rolling_sum({self._print(expr.args[1])})"

    def _print_ts_decay_linear(self, expr):
        return f"_ts_decay_linear({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_cs_rank(self, expr):
        # TODO: 此处最好有官方的解决方法
        return f"_rank_pct({self._print(expr.args[0])})"

    def _print_cs_scale(self, expr):
        return f"_scale({self._print(expr.args[0])})"

    def _print_log(self, expr):
        PREC = precedence(expr)
        if expr.args[0].is_Number:
            return f"np.log({expr.args[0]})"
        else:
            return f"{self.parenthesize(expr.args[0], PREC)}.log()"

    def _print_Abs(self, expr):
        PREC = precedence(expr)
        if expr.args[0].is_Number:
            return f"np.abs({expr.args[0]})"
        else:
            return f"{self.parenthesize(expr.args[0], PREC)}.abs()"

    def _print_Max(self, expr):
        return f"pl.max_horizontal([{self._print(expr.args[0])}, {self._print(expr.args[1])}])"

    def _print_Min(self, expr):
        return f"pl.min_horizontal([{self._print(expr.args[0])}, {self._print(expr.args[1])}])"

    def _print_sign(self, expr):
        PREC = precedence(expr)
        if expr.args[0].is_Number:
            return f"np.sign({expr.args[0]})"
        else:
            return f"{self.parenthesize(expr.args[0], PREC)}.sign()"

    def _print_signed_power(self, expr):
        # 太长了，所以这里简化一下
        return f"_signed_power({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_gp_rank(self, expr):
        return f"_rank_pct({self._print(expr.args[1])})"

    def _print_gp_neutralize(self, expr):
        return f"_neutralize({self._print(expr.args[1])})"
