import React, {Component} from 'react';
import Title from './Title';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from "highcharts-react-official";
import { withTheme } from '@material-ui/core/styles';

class CandleStickChart extends Component {

    constructor(props) {
        super(props);
        this.state = {
            chart: null
        }

    }

    componentDidMount() {
        console.log('REACT_APP_REST_API_URL -> ', process.env.REACT_APP_REST_API_URL);
        fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/bitmex/xbtusd/1d`)
            .then(res => res.json())
            .then((data) => {
                const ohlc_data = data['data'].sort(function(a, b) {
                    return a['timestamp']['epoch_time'] - b['timestamp']['epoch_time'];
                }).map(x =>[
                    x['timestamp']['epoch_time'] * 1000,
                    x['open'],
                    x['high'],
                    x['low'],
                    x['close']
                ]);

                const options = {
                    series : [{
                        type: 'candlestick',
                        name : 'XBTUSD',
                        color: this.props.theme.palette.secondary.light,
                        data : ohlc_data,
                    }],
                    rangeSelector: {
                        selected: 1
                    },
                }

                this.setState({chart: (<HighchartsReact
                        constructorType= { 'stockChart' }
                        highcharts={ Highcharts }
                        options = { options }
                    />)});

            })
            .catch(console.log)
    }

    render() {
        return (
            <React.Fragment>
                <Title>XBTUSD - Daily Data</Title>
                {this.state.chart}
            </React.Fragment>
        );
    }
}

export default withTheme(CandleStickChart);