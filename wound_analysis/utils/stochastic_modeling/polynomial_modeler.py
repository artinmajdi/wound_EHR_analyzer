"""
Polynomial modeling utilities for stochastic modeling.
"""

import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Optional, Tuple


class PolynomialModeler:
    """
    Handles polynomial regression modeling.
    """

    def __init__(self):
        """Initialize the polynomial modeler."""
        self.models = {}
        self.current_model = None

    def fit_models(self,
                  x: np.ndarray,
                  y: np.ndarray,
                  max_degree: int = 5,
                  criterion: str = 'aic') -> Dict[int, Dict]:
        """
        Fit polynomial models up to specified degree.

        Parameters:
        ----------
        x : np.ndarray
            Independent variable data
        y : np.ndarray
            Dependent variable data
        max_degree : int, optional
            Maximum polynomial degree to consider
        criterion : str, optional
            Model selection criterion ('aic' or 'bic')

        Returns:
        -------
        Dict[int, Dict]
            Dictionary of model results for each degree
        """
        results = {}
        x = x.reshape(-1, 1)

        for degree in range(1, max_degree + 1):
            # Create polynomial features
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(x)

            # Fit model
            model = LinearRegression()
            model.fit(X_poly, y)

            # Store model and features
            model.X = X_poly  # Store for later use
            model.y = y
            model.poly = poly
            self.models[degree] = model

            # Calculate metrics
            y_pred = model.predict(X_poly)
            residuals = y - y_pred
            n = len(y)
            p = degree + 1  # number of parameters

            # Calculate information criteria
            rss = np.sum(residuals ** 2)
            aic = n * np.log(rss/n) + 2 * p
            bic = n * np.log(rss/n) + np.log(n) * p

            # Calculate confidence bands
            confidence_bands = self._calculate_confidence_bands(
                model, X_poly, x, y_pred, residuals
            )

            # Store results
            results[degree] = {
                'model': model,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'r2': r2_score(y, y_pred),
                'adj_r2': 1 - (1 - r2_score(y, y_pred)) * (n - 1)/(n - p),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'aic': aic,
                'bic': bic,
                'residuals': residuals,
                'confidence_bands': confidence_bands,
                'degree': degree
            }

            # Add standard errors and p-values
            se, p_values = self._calculate_coefficient_stats(
                X_poly, y, model, residuals
            )
            results[degree]['std_errors'] = se
            results[degree]['p_values'] = p_values

        return results

    def get_best_model(self,
                      results: Dict[int, Dict],
                      criterion: str = 'aic') -> Dict:
        """
        Get the best model based on specified criterion.

        Parameters:
        ----------
        results : Dict[int, Dict]
            Dictionary of model results
        criterion : str, optional
            Model selection criterion ('aic' or 'bic')

        Returns:
        -------
        Dict
            Results for the best model
        """
        criterion = criterion.lower()
        if criterion not in ['aic', 'bic']:
            raise ValueError("Criterion must be 'aic' or 'bic'")

        # Find best model
        criterion_values = [results[d][criterion] for d in results]
        best_degree = list(results.keys())[np.argmin(criterion_values)]

        self.current_model = results[best_degree]['model']
        return results[best_degree]

    def predict(self,
               model,
               x: np.ndarray) -> np.ndarray:
        """
        Make predictions using a fitted model.

        Parameters:
        ----------
        model : object
            Fitted polynomial model
        x : np.ndarray
            Input values for prediction

        Returns:
        -------
        np.ndarray
            Predicted values
        """
        x = x.reshape(-1, 1)
        X_poly = model.poly.transform(x)
        return model.predict(X_poly)

    def _calculate_confidence_bands(self,
                                model,
                                X_poly: np.ndarray,
                                x: np.ndarray,
                                y_pred: np.ndarray,
                                residuals: np.ndarray,
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence bands for the polynomial fit.

        Parameters:
        ----------
        model : object
            Fitted model
        X_poly : np.ndarray
            Polynomial features matrix
        x : np.ndarray
            Original input values
        y_pred : np.ndarray
            Predicted values
        residuals : np.ndarray
            Model residuals
        confidence_level : float, optional
            Confidence level for bands

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper confidence bands
        """
        # Calculate standard error of the regression
        n = len(y_pred)
        p = len(model.coef_)
        mse = np.sum(residuals ** 2) / (n - p)

        # Calculate prediction variance
        X_new = model.poly.transform(x.reshape(-1, 1))
        leverage = np.diag(X_new @ np.linalg.inv(X_poly.T @ X_poly) @ X_new.T)
        pred_var = mse * (1 + leverage)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, n - p)
        margin = t_value * np.sqrt(pred_var)

        return y_pred - margin, y_pred + margin

    def _calculate_coefficient_stats(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  model,
                                  residuals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate standard errors and p-values for coefficients.

        Parameters:
        ----------
        X : np.ndarray
            Design matrix
        y : np.ndarray
            Dependent variable data
        model : object
            Fitted model
        residuals : np.ndarray
            Model residuals

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            Standard errors and p-values for coefficients
        """
        # Calculate standard errors
        n = len(y)
        p = len(model.coef_)
        mse = np.sum(residuals ** 2) / (n - p)
        var_coef = mse * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_coef))

        # Calculate t-statistics and p-values
        t_stats = model.coef_ / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))

        return se, p_values
