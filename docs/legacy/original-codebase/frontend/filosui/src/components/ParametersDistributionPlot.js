import React, {Component} from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from "highcharts-react-official";
import { withTheme } from '@material-ui/core/styles';
import bellcurve from 'highcharts/modules/histogram-bellcurve';

bellcurve(Highcharts);

class ParametersDistributionPlot extends Component {

    constructor(props) {
        super(props);
        this.state = {
            chart: null,
        }
    }


    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevProps.tid !== this.props.tid){
            this.componentDidMount();
        }
    }

    componentDidMount() {
        const options = {
            title: {
                text: this.props.chartTitle
            },
            xAxis: [
                {
                    title: {
                        text: this.props.xLabel
                    },
                    alignTicks: false
                }, {
                    title: {
                        text: 'Bell curve'
                    },
                    alignTicks: false,
                    opposite: true,
                    plotLines: [{
                        label: this.props.vLineLabel,
                        color: this.props.theme.palette.warning.main,
                        width: 3,
                        value: this.props.vLineValue
                    }]

                }],
            yAxis: [
                {
                    title: { text: this.props.yLabel }
                }, {
                    title: { text: 'Bell curve' },
                    opposite: true
                }],            series: [{
                name: 'Bell curve',
                type: 'bellcurve',
                xAxis: 1,
                yAxis: 1,
                baseSeries: 1,
                zIndex: -10,
                opacity: 0.3,
                color: this.props.theme.palette.secondary.light
            }, {
                name: this.props.yLabel,
                type: 'scatter',
                data: this.props.data,
                color: this.props.theme.palette.primary.main,
                accessibility: {
                    exposeAsGroupOnly: true
                },
                marker: {
                    radius: 3.5
                }
            }]
        };

        this.setState({chart: (<HighchartsReact
                highcharts={ Highcharts }
                options = { options }
            />)});
    }

    render() {
        return (
            <React.Fragment>
                {this.state.chart}
            </React.Fragment>
        );
    }
}

export default withTheme(ParametersDistributionPlot);