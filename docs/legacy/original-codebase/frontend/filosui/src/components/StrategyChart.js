import React, {Component} from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from "highcharts-react-official";
import { withTheme } from '@material-ui/core/styles';

class StrategyChart extends Component {

    constructor(props) {
        super(props);
        this.state = {
            chart: undefined
        }

    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevProps.data !== this.props.data){
            this.componentDidMount();
        }
    }

    componentDidMount() {
        const indicators = this.props.data['indicators']
        fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${this.props.provider}/${this.props.symbol}/1d/${this.props.startDate}/${this.props.endDate}`)
            .then(res => res.json())
            .then((ohlc_data) => {
                ohlc_data = ohlc_data['data'].sort(function (a, b) {
                    return a['timestamp']['epoch_time'] - b['timestamp']['epoch_time'];
                }).map(x => [
                    x['timestamp']['epoch_time'] * 1000,
                    x['open'],
                    x['high'],
                    x['low'],
                    x['close']
                ]);
                
                const options = {
                    rangeSelector:{
                        selected: 1,
                    },
                    series : [
                        {
                            id: 'dataseries',
                            type: 'candlestick',
                            data : ohlc_data,
                            color: this.props.theme.palette.secondary.light
                        },
                        {
                            type: 'flags',
                            data: this.props.data['buy'].sort(function(a, b) {return a['x'] - b['x']}),
                            onSeries: 'dataseries',
                            shape: 'squarepin',
                            width: 16,
                            color: this.props.theme.palette.info.main,
                            style: {
                                color: this.props.theme.palette.info.main
                            },
                        }, {
                            type: 'flags',
                            data: this.props.data['sell'].sort(function(a, b) {return a['x'] - b['x']}),
                            onSeries: 'dataseries',
                            shape: 'circlepin',
                            width: 16,
                            color: this.props.theme.palette.error.main,
                            style: {
                                color: this.props.theme.palette.error.main
                            },
                        }
                    ],
                }

                for (let indicator in indicators){
                    for (let line in indicators[indicator]){
                        options['series'].push({
                            name: [indicators[indicator][line]['label'],
                                   indicators[indicator][line]['name']].join(' - '),
                            data: ohlc_data.map(function(e, i ){
                                return [e[0], indicators[indicator][line]['x'][i]]})
                        })
                    }
                }
                

                this.setState({chart: (<HighchartsReact
                        constructorType= { 'stockChart' }
                        highcharts={ Highcharts }
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

export default withTheme(StrategyChart);