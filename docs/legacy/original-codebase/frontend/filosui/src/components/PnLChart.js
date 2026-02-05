import React, {Component} from 'react';
import Highcharts from 'highcharts/highstock';
import HighchartsReact from "highcharts-react-official";
import { withTheme } from '@material-ui/core/styles';

class PnLChart extends Component {

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
        let pnl_history = this.props.data;

        const options = {
            chart: {
                type: 'line',
            },
            title: null,
            subTitle: null,
            xAxis: {
                title: {
                    text: 'Trade'
                },
            },
            yAxis: {
                title: {
                    text: 'PNL'
                },
            },
            series: [{
                name: 'Pnl',
                color: this.props.theme.palette.secondary.light,
                data: pnl_history
            }]};

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

export default withTheme(PnLChart);