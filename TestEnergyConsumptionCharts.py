import unittest
import matplotlib.pyplot as plt

# Import the class to be tested
from energy_consumption_chart import EnergyConsumptionChart

class TestEnergyConsumptionChart(unittest.TestCase):
    def test_plot_chart(self):
        renewable_data = [50, 60, 70, 80, 90]
        non_renewable_data = [150, 140, 130, 120, 110]
        years = [2015, 2016, 2017, 2018, 2019]

        chart = EnergyConsumptionChart(renewable_data, non_renewable_data, years)
        chart.plot_chart()

        # Assert that the chart is displayed
        self.assertTrue(plt.gcf().number > 0)

        # Optional: Save the chart as an image file for visual inspection
        # plt.savefig('energy_consumption_chart.png')

        # Optional: Close the chart to prevent it from blocking subsequent tests
        # plt.close()

if __name__ == '__main__':
    unittest.main()
