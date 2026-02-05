import React, {Component} from 'react';
import {withStyles} from '@material-ui/core/styles';
import TableContainer from "@material-ui/core/TableContainer";
import Table from "@material-ui/core/Table";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import TableCell from "@material-ui/core/TableCell";
import TableBody from "@material-ui/core/TableBody";
import IconButton from "@material-ui/core/IconButton";
import KeyboardArrowUpIcon from "@material-ui/icons/KeyboardArrowUp";
import KeyboardArrowDownIcon from "@material-ui/icons/KeyboardArrowDown";
import Box from "@material-ui/core/Box";
import Typography from "@material-ui/core/Typography";
import {createMuiTheme} from "@material-ui/core";
import Grid from "@material-ui/core/Grid";
import ParametersDistributionPlot from "./ParametersDistributionPlot";
import Collapse from "@material-ui/core/Collapse";


const theme = createMuiTheme();

const styles = {
    paper: {
        padding: theme.spacing(2),
        display: 'flex',
        overflow: 'auto',
        flexDirection: 'column',
    },
    root: {
        '& > *': {
            borderBottom: 'unset',
        },
    },
};

class WalkForwardMetrics extends Component {

    constructor(props) {
        super(props);
        this.state = {
            stats: []
        }
        this.showMetricDistribution = this.showMetricDistribution.bind(this);
        this.getWalkForwardsStatsComponent = this.getWalkForwardsStatsComponent.bind(this);
        this.computeData = this.computeData.bind(this);
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        const prevTid = Object.keys(prevProps.data).length === 0 ? "" : prevProps.data['tid'];
        if (prevTid !== this.props.data['tid']) {
            this.computeData();
        }
    }

    computeData(){
        if (Object.keys(this.props.data).length === 0 ||
            this.props.data['optimizations']['WALKFORWARD'].length === 0) {
            return
        }

        const stats = this.computeWalkForwardsStats(this.props.data['optimizations']['WALKFORWARD']);
        const newState = this.state;

        for (let i = 0; i < stats.length; i++) {
            newState[stats[i][0]] = false;
        }

        newState['tid'] = this.props.data['tid'];
        newState['stats'] = stats;
        this.setState(newState);
    }

    componentDidMount() {
        this.computeData();
    }

    getWalkForwardsStatsComponent(){
        if (this.state.stats.length === 0) {
            return this.getNoDataMsg();
        }

        const {classes} = this.props;
        return (<TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell/>
                            <TableCell>Metric</TableCell>
                            <TableCell>Value</TableCell>
                            <TableCell>Std. Dev.</TableCell>
                            <TableCell>$</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {
                            this.state.stats.map(m => (
                                <React.Fragment key={m[0] + "-rf"} >
                                    <TableRow key={m[0]} hover className={classes.root}>
                                        <TableCell>
                                            <IconButton
                                                aria-label="expand row"
                                                size="small"
                                                onClick={() => this.showMetricDistribution.bind(this)(m[0])}
                                            >
                                                {this.state[m[0]] ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                                            </IconButton>
                                        </TableCell>
                                        <TableCell>{m[0]}</TableCell>
                                        <TableCell>
                                            {this.getMetricValue(m[0], m[1])}
                                        </TableCell>
                                        <TableCell>
                                            &#177; {m[2] !== null ? m[2].toFixed(3) : "N/A"}</TableCell>
                                        <TableCell>
                                            {this.getMetricDollars(m[0], m[1], this.props.data['cash'])}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={6}>
                                            <Collapse in={this.state[m[0]]}
                                                      key={m[0] + 'c'}
                                                      name={m[0] + 'c'} timeout="auto" unmountOnExit>
                                                <Box margin={1}>
                                                    <Typography variant="h6" gutterBottom component="div">
                                                        {m[0]} Distribution
                                                    </Typography>
                                                    {this.getWalkForwardMetricsDistribution(m[0],
                                                        this.props.data)}
                                                </Box>
                                            </Collapse>
                                        </TableCell>
                                    </TableRow>

                                </React.Fragment>
                            ))
                        }
                    </TableBody>
                </Table>
            </TableContainer>
        )
    }

    showMetricDistribution(metric) {
        this.setState({[metric]: !this.state[metric]});
    }

    computeWalkForwardsStats(datumElement) {
        let metricArrays = {};

        for (let i = 0; i < datumElement.length; i++) {
            for (const key in datumElement[i]["analyzers"]["PyFolio"]) {
                if (!metricArrays.hasOwnProperty(key)) {
                    metricArrays[key] = []
                }
                metricArrays[key].push(datumElement[i]["analyzers"]["PyFolio"][key])
            }
        }

        let stats = Object.keys(metricArrays).map(k => {
            let mean = this.computeMean(metricArrays[k]);
            let std = this.computeStd(metricArrays[k], mean);
            return [k, mean, std]
        });

        return stats;
    }

    computeMean(data) {
        const sum = data.reduce(function (sum, value) {
            return sum + value;
        }, 0);

        return sum / data.length;
    }

    computeStd(values, mean) {

        const squareDiffs = values.map(function (value) {
            const diff = value - mean;
            return diff * diff;
        });

        const meanSquareDiff = this.computeMean(squareDiffs);
        return Math.sqrt(meanSquareDiff);
    }

    getNoDataMsg() {
        return <p>No Data Available</p>
    }

    getWalkForwardMetricsDistribution(metric, data) {
        const optimizations = data["optimizations"];
        if (optimizations["WALKFORWARD"].length === 0) {
            return <p>No Data Available. </p>
        }

        const metrics = {};

        const walkForwardData = optimizations["WALKFORWARD"].sort(function (a, b) {
            return a["num_split"] < b["num_split"] ? -1 :  a["num_split"] > b["num_split"] ? 1 : 0;
        });

        for (let i = 0; i < walkForwardData.length; i++) {
            let analyzerDict = walkForwardData[i]["analyzers"]["PyFolio"];

            Object.keys(analyzerDict).filter(m=> m === metric).forEach(a => {
                if (!metrics.hasOwnProperty(a)){
                    metrics[a] = []
                }
                metrics[a].push(analyzerDict[a]);
            })
        }


        return (
            <Grid container>
                {Object.keys(metrics).map(m=>
                    <Grid item xs={12} key={m} >
                        <ParametersDistributionPlot
                            chartTitle={""}
                            xLabel={'Out of Sample'}
                            yLabel={'Metric'}
                            vLineLabel={'Backtesting'}
                            vLineValue={data["optimizations"]["BACKTESTING"][0]["analyzers"]["PyFolio"][m]}
                            data={metrics[m]}/>
                    </Grid>)}
            </Grid>
        );
    }

    render() {
        return (this.getWalkForwardsStatsComponent())
    }

    getMetricValue(metricName, metricValue){
        if (metricValue === null ){
            return "N/A";
        }

        const percMetrics = new Set([
            'Annual return',
            'Annual volatility',
            'Cumulative returns',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        ]);


        return percMetrics.has(metricName) ? `${(metricValue * 100.0).toFixed(3)} %` :
            metricValue.toFixed(3);
    }

    getMetricDollars(metric, value, cash){
        if (value === null ){
            return "N/A";
        }

        const percMetrics = new Set([
            'Annual return',
            'Annual volatility',
            'Cumulative returns',
            'Max drawdown',
            'Daily value at risk',
            'Daily turnover'
        ]);

        const formatter = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
        });

        return percMetrics.has(metric) ?
            formatter.format(value * cash):"";
    }
}

export default withStyles(styles)(WalkForwardMetrics);