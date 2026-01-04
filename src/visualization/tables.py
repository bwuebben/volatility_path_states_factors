"""
LaTeX table generation for the paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TableGenerator:
    """
    Generate LaTeX tables for the paper.
    
    Parameters
    ----------
    output_dir : str, default 'output/tables'
        Directory to save tables.
    decimal_places : int, default 2
        Default decimal places for formatting.
        
    Examples
    --------
    >>> generator = TableGenerator()
    >>> generator.table1_summary_statistics(factor_returns, regimes)
    >>> generator.table2_state_conditional_returns(factor_returns, regimes)
    """
    
    REGIME_ORDER = [
        'Calm Trend',
        'Choppy Transition',
        'Slow-Burn Stress',
        'Crash-Spike',
        'Recovery',
    ]
    
    def __init__(
        self,
        output_dir: str = 'output/tables',
        decimal_places: int = 2,
    ):
        """Initialize table generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decimal_places = decimal_places
        
    def save_table(self, latex: str, name: str):
        """Save LaTeX table to file."""
        path = self.output_dir / f"{name}.tex"
        with open(path, 'w') as f:
            f.write(latex)
        logger.info(f"Saved: {path}")
    
    def _format_number(
        self,
        x: float,
        decimals: int = None,
        pct: bool = False,
    ) -> str:
        """Format a number for LaTeX."""
        if decimals is None:
            decimals = self.decimal_places
        
        if pd.isna(x):
            return '--'
        
        if pct:
            return f'{x:.{decimals}f}\\%'
        return f'{x:.{decimals}f}'
    
    def _format_significance(self, pvalue: float) -> str:
        """Add significance stars."""
        if pvalue < 0.01:
            return '***'
        elif pvalue < 0.05:
            return '**'
        elif pvalue < 0.10:
            return '*'
        return ''
    
    def table1_summary_statistics(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        save: bool = True,
    ) -> str:
        """
        Generate Table 1: Summary Statistics.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        str
            LaTeX table code.
        """
        # Compute statistics
        factors = list(factor_returns.columns)
        
        stats = []
        for factor in factors:
            ret = factor_returns[factor].dropna()
            stats.append({
                'Factor': factor,
                'Mean': ret.mean() * 12 * 100,
                'Std': ret.std() * np.sqrt(12) * 100,
                'Sharpe': ret.mean() / ret.std() * np.sqrt(12),
                'Skew': ret.skew(),
                'Kurt': ret.kurtosis(),
                'Min': ret.min() * 100,
                'Max': ret.max() * 100,
            })
        
        df = pd.DataFrame(stats).set_index('Factor')
        
        # Build LaTeX
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Summary Statistics: Factor Returns}
\label{tab:summary_stats}
\begin{tabular}{lrrrrrrr}
\toprule
& \textbf{Mean} & \textbf{Std} & \textbf{Sharpe} & \textbf{Skew} & \textbf{Kurt} & \textbf{Min} & \textbf{Max} \\
\midrule
"""
        
        for factor in factors:
            row = df.loc[factor]
            latex += f"{factor} & {row['Mean']:.2f} & {row['Std']:.2f} & {row['Sharpe']:.2f} & "
            latex += f"{row['Skew']:.2f} & {row['Kurt']:.2f} & {row['Min']:.2f} & {row['Max']:.2f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}

\medskip
\footnotesize
\textit{Notes:} Mean and Std are annualized and in percent. Min and Max are monthly returns in percent.
\end{table}
"""
        
        if save:
            self.save_table(latex, 'table1_summary_stats')
        
        return latex
    
    def table2_state_conditional_returns(
        self,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
        save: bool = True,
    ) -> str:
        """
        Generate Table 2: State-Conditional Factor Returns.
        
        Parameters
        ----------
        factor_returns : pd.DataFrame
            Factor returns.
        regimes : pd.Series
            Regime classification.
        save : bool, default True
            Whether to save.
            
        Returns
        -------
        str
            LaTeX table code.
        """
        # Align data
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        factors = list(returns.columns)
        
        # Compute statistics by regime
        latex = r"""
\begin{table}[htbp]
\centering
\caption{State-Conditional Factor Returns}
\label{tab:state_conditional}
\begin{tabular}{l""" + 'r' * len(factors) + r"""}
\toprule
& """ + ' & '.join([f'\\textbf{{{f}}}' for f in factors]) + r""" \\
\midrule
\multicolumn{""" + str(len(factors) + 1) + r"""}{l}{\textit{Mean Monthly Return (\%)}} \\
\addlinespace
"""
        
        for regime in self.REGIME_ORDER:
            mask = reg == regime
            if not mask.any():
                continue
            
            values = []
            for factor in factors:
                mean = returns.loc[mask, factor].mean() * 100
                values.append(f'{mean:.2f}')
            
            latex += f"{regime} & " + ' & '.join(values) + " \\\\\n"
        
        latex += r"""
\addlinespace
\midrule
\multicolumn{""" + str(len(factors) + 1) + r"""}{l}{\textit{Sharpe Ratio}} \\
\addlinespace
"""
        
        for regime in self.REGIME_ORDER:
            mask = reg == regime
            if not mask.any():
                continue
            
            values = []
            for factor in factors:
                ret = returns.loc[mask, factor]
                sr = ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0
                values.append(f'{sr:.2f}')
            
            latex += f"{regime} & " + ' & '.join(values) + " \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}

\medskip
\footnotesize
\textit{Notes:} Table reports mean monthly returns and annualized Sharpe ratios by path state regime.
\end{table}
"""
        
        if save:
            self.save_table(latex, 'table2_state_conditional')
        
        return latex
    
    def table3_regime_frequencies(
        self,
        regimes: pd.Series,
        save: bool = True,
    ) -> str:
        """
        Generate Table 3: Regime Frequencies and Characteristics.
        """
        # Compute frequencies
        freq = regimes.value_counts()
        pct = freq / len(regimes) * 100
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Regime Frequencies and Transition Probabilities}
\label{tab:regime_freq}
\begin{tabular}{lrr}
\toprule
\textbf{Regime} & \textbf{Observations} & \textbf{Frequency (\%)} \\
\midrule
"""
        
        for regime in self.REGIME_ORDER:
            n = freq.get(regime, 0)
            p = pct.get(regime, 0)
            latex += f"{regime} & {n} & {p:.1f} \\\\\n"
        
        latex += r"""
\midrule
Total & """ + str(len(regimes)) + r""" & 100.0 \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        if save:
            self.save_table(latex, 'table3_regime_freq')
        
        return latex
    
    def table4_portfolio_performance(
        self,
        baseline_returns: pd.Series,
        conditioned_returns: pd.Series,
        vol_scaled_returns: pd.Series = None,
        save: bool = True,
    ) -> str:
        """
        Generate Table 4: Portfolio Performance Comparison.
        """
        def compute_stats(ret):
            return {
                'mean': ret.mean() * 12 * 100,
                'vol': ret.std() * np.sqrt(12) * 100,
                'sharpe': ret.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else 0,
                'skew': ret.skew(),
                'max_dd': self._compute_max_drawdown(ret) * 100,
            }
        
        strategies = {
            'Unconditional': compute_stats(baseline_returns),
            'State-Conditioned': compute_stats(conditioned_returns),
        }
        
        if vol_scaled_returns is not None:
            strategies['Vol-Scaled'] = compute_stats(vol_scaled_returns)
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Portfolio Performance Comparison}
\label{tab:portfolio_perf}
\begin{tabular}{l""" + 'r' * len(strategies) + r"""}
\toprule
& """ + ' & '.join([f'\\textbf{{{s}}}' for s in strategies.keys()]) + r""" \\
\midrule
"""
        
        metrics = [
            ('Mean Return (\\%)', 'mean'),
            ('Volatility (\\%)', 'vol'),
            ('Sharpe Ratio', 'sharpe'),
            ('Skewness', 'skew'),
            ('Max Drawdown (\\%)', 'max_dd'),
        ]
        
        for label, key in metrics:
            values = [f"{strategies[s][key]:.2f}" for s in strategies.keys()]
            latex += f"{label} & " + ' & '.join(values) + " \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}

\medskip
\footnotesize
\textit{Notes:} Mean return and volatility are annualized.
\end{table}
"""
        
        if save:
            self.save_table(latex, 'table4_portfolio_perf')
        
        return latex
    
    def table5_crash_analysis(
        self,
        factor_returns: pd.Series,
        regimes: pd.Series,
        crash_percentile: float = 5,
        save: bool = True,
    ) -> str:
        """
        Generate Table 5: Crash Episode Analysis.
        """
        # Align data
        common_idx = factor_returns.index.intersection(regimes.index)
        returns = factor_returns.loc[common_idx]
        reg = regimes.loc[common_idx]
        
        # Identify crashes
        threshold = returns.quantile(crash_percentile / 100)
        crashes = returns <= threshold
        
        # Crashes by regime
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Crash Episode Analysis}
\label{tab:crash_analysis}
\begin{tabular}{lrrr}
\toprule
\textbf{Regime} & \textbf{Total Obs} & \textbf{Crashes} & \textbf{Crash Prob (\%)} \\
\midrule
"""
        
        for regime in self.REGIME_ORDER:
            mask = reg == regime
            n_total = mask.sum()
            n_crash = (crashes & mask).sum()
            prob = n_crash / n_total * 100 if n_total > 0 else 0
            
            latex += f"{regime} & {n_total} & {n_crash} & {prob:.1f} \\\\\n"
        
        # Overall
        n_crash_total = crashes.sum()
        prob_total = crash_percentile
        
        latex += r"""
\midrule
Overall & """ + str(len(returns)) + f" & {n_crash_total} & {prob_total:.1f}" + r""" \\
\bottomrule
\end{tabular}

\medskip
\footnotesize
\textit{Notes:} Crashes defined as returns below the """ + f"{crash_percentile}th" + r""" percentile.
\end{table}
"""
        
        if save:
            self.save_table(latex, 'table5_crash_analysis')
        
        return latex
    
    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """Compute maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return drawdown.max()
    
    def generate_all_tables(
        self,
        data: Dict,
    ):
        """
        Generate all paper tables.
        """
        logger.info("Generating all tables...")
        
        factors = data['factors']
        regimes = data.get('regimes')
        if isinstance(regimes, pd.DataFrame):
            regimes = regimes['regime']
        
        self.table1_summary_statistics(factors, regimes)
        self.table2_state_conditional_returns(factors, regimes)
        self.table3_regime_frequencies(regimes)
        
        if 'Momentum' in factors.columns:
            self.table5_crash_analysis(factors['Momentum'], regimes)
        
        logger.info("All tables generated successfully!")
