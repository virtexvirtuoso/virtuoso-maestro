import React, {Component} from 'react';
import Highcharts from 'highcharts';
import Heatmap from 'highcharts/modules/heatmap.js';
import Exporting from 'highcharts/modules/exporting';
import Data from 'highcharts/modules/data'
import CanvasBoost from 'highcharts/modules/boost-canvas';
import Boost from 'highcharts/modules/boost';
import Accessibility from 'highcharts/modules/accessibility'
import HighchartsReact from "highcharts-react-official";
import { withTheme } from '@material-ui/core/styles';

Accessibility(Highcharts);
Data(Highcharts);
Exporting(Highcharts);
Boost(Highcharts);
CanvasBoost(Highcharts);
Heatmap(Highcharts);

class HeatMapChart extends Component {

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
        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${this.props.tid}/correlation`)
            .then(response => response.json())
            .then(data => {
                const options = {

                    data: {
                        csv: data['csv']
                    },

                    chart: {
                        type: 'heatmap'
                    },

                    boost: {
                        useGPUTranslations: true
                    },

                    title: {
                        text: '',
                        align: 'left',
                        x: 40
                    },

                    subtitle: {
                        text: 'Daily Buy/Sell Signal - Baktesting vs WalkForward',
                        align: 'left',
                        x: 40
                    },

                    xAxis: {
                        type: 'datetime',
                        min: data['min_ts'] * 1000,
                        max: data['max_ts'] * 1000,
                        labels: {
                            align: 'left',
                            x: 5,
                            y: 14,
                            format: '{value:%B}' // long month
                        },
                        showLastLabel: false,
                        tickLength: 16
                    },

                    yAxis: {
                        title: {
                            text: null
                        },
                        labels: {
                            format: '{value}'
                        },
                        minPadding: 0,
                        maxPadding: 0,
                        startOnTick: false,
                        endOnTick: false,
                        tickPositions: data['tests'],
                        tickWidth: 1,
                        min: Math.min(...data['tests']),
                        max: Math.max(...data['tests']),
                        reversed: true
                    },

                    colorAxis: {
                        stops: [
                            [0, '#3060cf'],
                            [0.5, '#fffbbc'],
                            [0.9, '#c4463a'],
                            [1, '#c4463a']
                        ],
                        min: data['min_value'],
                        max: data['max_value'],
                        startOnTick: false,
                        endOnTick: false,
                        labels: {
                            format: '{value}'
                        }
                    },

                    series: [{
                        boostThreshold: 100,
                        borderWidth: 0,
                        nullColor: '#EFEFEF',
                        colsize: 24 * 36e5, // one day
                        tooltip: {
                            headerFormat: 'Buy/Sell Signal<br/>',
                            pointFormat: '{point.x:%e %b, %Y} {point.y} <b>{point.value}</b>'
                        },
                        turboThreshold: Number.MAX_SAFE_INTEGER // #3404, remove after 4.0.5 release
                    }]

                }

                this.setState({chart: (<HighchartsReact
                        highcharts = {Highcharts}
                        options = { options }
                    />)});
            })


    }

    render() {
        return (
            <React.Fragment>
                {this.state.chart}
            </React.Fragment>
        );
    }
}

export default withTheme(HeatMapChart);