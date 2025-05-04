from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd


class RiskAwareStockTradingEnv(StockTradingEnv):
    def __init__(self, df, **kwargs):
        self.risk_score_col = 'risk_score'
        super().__init__(df, **kwargs)

    def step(self, actions):
        state, reward, done, truncated, info = super().step(actions)

        # Get the risk score
        day_idx = max(0, self.day - 1)
        try:
            risk_score = self.df.loc[day_idx, self.risk_score_col]
            risk_score = risk_score if pd.notnull(risk_score) else 0
        except KeyError:
            risk_score = 0

        # Inject risk into state indirectly (scale the cash balance)
        risk_weight = self._get_risk_scaling_factor(risk_score)

        # Assume state[0] = cash balance
        state[0] = state[0] * risk_weight

        return state, reward, done, truncated, info

    def _get_risk_scaling_factor(self, risk_score):
        if risk_score == 0:
            return 1.0  # No change
        elif 1 <= risk_score <= 2:
            return 1.3  # More optimistic
        elif risk_score == 3:
            return 1.1  # Slight optimistic
        elif 4 <= risk_score <= 5:
            return 0.8  # Slight caution
        else:
            return 1.0
