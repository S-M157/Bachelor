from typing import List, Union, Callable, Dict, Optional

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


class DriftCalculator:
    def __init__(self, drift_metrics: List[Union[str, Callable]], report_names: List[str]):
        self.drift_stat_tests = drift_metrics
        self.reports: List[Report] = []
        self.reports_names = report_names
        self.reference = pd.DataFrame()
        self.fi: Optional[Dict[str, float]] = None

        self._renew_reports()

    def _renew_reports(self):
        self.reports: List[Report] = [Report(metrics=[DataDriftPreset(num_stattest=t),]) for t in self.drift_stat_tests]

    def set_reference(self, new_ref: pd.DataFrame) -> None:
        self.reference = new_ref

    def set_fi(self, fi: Dict[str, float]):
        self.fi = fi

    def get_drift(self,
                  current_data: pd.DataFrame,
                  reference_data: Optional[pd.DataFrame] = None,
                  sample: bool = False,
                  weighted: bool = False,
                  save_plot: Optional[str] = None) -> Dict[str, float]:
        reff = self.reference

        if reference_data is not None:
            reff = reference_data

        scores = {}
        for report, report_name in zip(self.reports, self.reports_names):
            n_ref = len(reff)
            n_cur = len(current_data)

            if sample:
                n = min(len(current_data), len(reff), 1000)
                n_ref = n
                n_cur = n

            report.run(reference_data=reff.sample(n=n_ref), current_data=current_data.sample(n=n_cur),)

            if save_plot:
                report.save_html(f'{save_plot}.html')

            drift_score = report.as_dict()['metrics'][0]['result']['share_of_drifted_columns']

            drift_statuses = {k: v['drift_detected'] for k, v in
                              report.as_dict()['metrics'][1]['result']['drift_by_columns'].items()
                              }

            if weighted:
                drift_score = 0
                for column, status in drift_statuses:
                    drift_score += int(status) * self.fi[column]
                drift_score /= sum(self.fi.values())

            scores[report_name] = drift_score

        self._renew_reports()

        return scores
