import React from 'react';
import 'fontsource-roboto';
import {withStyles} from '@material-ui/core/styles';
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import Title from "./Title";
import {createMuiTheme} from "@material-ui/core";
import Table from "@material-ui/core/Table";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableCell from "@material-ui/core/TableCell";
import TableRow from "@material-ui/core/TableRow";
import TableBody from "@material-ui/core/TableBody";
import ListItem from "@material-ui/core/ListItem";
import ListItemText from "@material-ui/core/ListItemText";
import List from "@material-ui/core/List";
import TextFieldsIcon from '@material-ui/icons/TextFields';
import AccessTimeIcon from '@material-ui/icons/AccessTime';
import ListItemAvatar from "@material-ui/core/ListItemAvatar";
import Avatar from "@material-ui/core/Avatar";
import MonetizationOnIcon from '@material-ui/icons/MonetizationOn';
import TimerIcon from '@material-ui/icons/Timer';
import StrategyChart from "./StrategyChart";
import PnLChart from "./PnLChart";
import AccountBalanceIcon from '@material-ui/icons/AccountBalance';
import MoneyOffIcon from '@material-ui/icons/MoneyOff';
import ParametersDistributionPlot from "./ParametersDistributionPlot";
import IconButton from "@material-ui/core/IconButton";
import KeyboardArrowDownIcon from '@material-ui/icons/KeyboardArrowDown';
import WalkForwardMetrics from "./WalkForwardMetrics";
import HeatMapChart from "./HeatMap";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import DateRangeIcon from '@material-ui/icons/DateRange';
import BusinessIcon from '@material-ui/icons/Business';
import Button from "@material-ui/core/Button";
import Link from "@material-ui/core/Link";

const theme = createMuiTheme();

const styles = {
    paper: {
        padding: theme.spacing(2),
        display: 'flex',
        overflow: 'auto',
        flexDirection: 'column',
    },
    treeView: {
        height: 240,
        flexGrow: 1,
        maxWidth: 400,
    },
    formControl: {
        margin: 5,
        minWidth: 120,
    },
};

class Evaluation extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            tid: "",
            available_tests: [],
            results: {}
        }
        this.handleInputChange = this.handleInputChange.bind(this);
    }

    componentDidMount() {
        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/available`)
            .then(response => response.json())
            .then(available_tests => {
                this.setState({
                        available_tests: available_tests,
                    },
                    () => {
                        if (this.props.match.params.tid !== undefined) {
                            this.setState({tid: this.props.match.params.tid}, this.fetchResults)
                        }
                    })
            })
    }

    render() {
        const { classes } = this.props;
        return (
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Tests</Title>
                        {
                            <FormControl className={classes.formControl} >
                                <InputLabel id="test-select-label">Test</InputLabel>
                                <Select
                                    labelId="test-select-label"
                                    id="test-select"
                                    name="test"
                                    value={this.state.tid}
                                    onChange={this.handleInputChange}
                                >
                                    {
                                        this.state.available_tests.sort(function(a,b) {
                                            return a['test_name'] < b['test_name'] ? -1 : 1;
                                        }).map(row => (
                                            <MenuItem key={row['tid']} value={row['tid']}>
                                                {row['test_name']}
                                            </MenuItem>))
                                    }
                                </Select>
                            </FormControl>
                        }
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Strategy Evaluation</Title>
                        {this.getStrategyDetails()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Backtest Strategy Chart</Title>
                        {this.getBacktestStrategyChart()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Walkforward Strategy Chart</Title>
                        {this.getWalkForwardStrategyChart()}
                    </Paper>
                </Grid>
                <Grid item xs={6}>
                    <Paper className={classes.paper}>
                        <Title>Backtesting</Title>
                        {this.getBacktestMetrics()}
                    </Paper>
                </Grid>
                <Grid item xs={6}>
                    <Paper className={classes.paper}>
                        <Title>Walk Forward</Title>
                        {this.getWalkForwardMetrics()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Backtest PnL Chart</Title>
                        {this.getBacktestPnLChart()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Walkforward PnL Chart</Title>
                        {this.getWalkForwardPnLChart()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Parameters Distribution</Title>
                        {this.getParametersDistribution()}
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Heatmap Chart</Title>
                        {this.getHeatMapChart()}
                    </Paper>
                </Grid>
            </Grid>);
    }

    getBacktestMetrics() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const datumElement = this.state.results['optimizations']['BACKTESTING'];

        if (datumElement.length === 0) {
            return this.getNoDataMsg();
        }

        let pyfolioMetrics = datumElement[0]['analyzers']['PyFolio'];

        return (
            <TableContainer>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell/>
                            <TableCell>Metric</TableCell>
                            <TableCell>Value</TableCell>
                            <TableCell>$</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {
                            Object.keys(pyfolioMetrics)
                                .map(m => (
                                    <TableRow key={m} hover>
                                        <TableCell>
                                            <IconButton
                                                aria-label="expand row"
                                                size="small"
                                                disabled>
                                                <KeyboardArrowDownIcon />
                                            </IconButton>
                                        </TableCell>
                                        <TableCell>{m}</TableCell>
                                        <TableCell>{this.getMetricValue(m, pyfolioMetrics)}</TableCell>
                                        <TableCell>{this.getMetricDollars(m, pyfolioMetrics,
                                            this.state.results['cash'])}</TableCell>
                                    </TableRow>))
                        }
                    </TableBody>
                </Table>
            </TableContainer>
        )
    }

    getMetricValue(metric, pyfolioMetrics){
        if (pyfolioMetrics[metric] === null ){
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


        return percMetrics.has(metric) ? `${(pyfolioMetrics[metric] * 100.0).toFixed(3)} %` :
            pyfolioMetrics[metric].toFixed(3);
    }

    getMetricDollars(metric, pyfolioMetrics, cash){
        if (pyfolioMetrics[metric] === null ){
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
            formatter.format(pyfolioMetrics[metric] * cash):"";
    }

    getStrategyDetails() {
        // Create our number formatter.
        const formatter = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
        });

        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        return (
            <List dense={true}>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <TextFieldsIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={this.state.results['test_name']} secondary="Test Name" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <AccessTimeIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                        primary={new Date(this.state.results['creation_time']).toLocaleString()}
                        secondary="Creation Time" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <DateRangeIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText
                        primary={<span>
                            From
                            <b> {new Date(this.state.results['start_date']).toLocaleString()} </b>
                            To
                            <b> {new Date(this.state.results['end_date']).toLocaleString()} </b>
                        </span>
                        }
                        secondary="Time Interval" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <BusinessIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={this.state.results['provider']} secondary="Provider" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <AccountBalanceIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={this.state.results['symbol']} secondary="Symbol" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <TimerIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={this.state.results['timeframe']} secondary="Timeframe" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <MonetizationOnIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={formatter.format(this.state.results['cash'])} secondary="Cash" />
                </ListItem>
                <ListItem>
                    <ListItemAvatar>
                        <Avatar>
                            <MoneyOffIcon />
                        </Avatar>
                    </ListItemAvatar>
                    <ListItemText primary={this.state.results['commissions'] + "%"} secondary="Commissions" />
                </ListItem>
                <ListItem>
                    <Link
                        href={`${process.env.REACT_APP_REST_API_URL}/optimization/results/${this.state.tid}/report`}
                        target="_blank" rel="noopener noreferrer" download underlineNone
                    >
                        <Button variant="contained" color="primary" component="span">Export</Button>
                    </Link>
                </ListItem>

            </List>
        );
    }

    getWalkForwardMetrics() {
        return (<WalkForwardMetrics tid={this.state.tid} data={this.state.results} />)
    }

    getNoDataMsg() {
        return <p>No Data Available</p>
    }

    getBacktestStrategyChart() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const datumElement = this.state.results['optimizations']['BACKTESTING'];

        if (datumElement.length === 0) {
            return this.getNoDataMsg();
        }

        let buy = datumElement[0]["observers"]["BuySell"]['buy'].map(e => {
            return {
                "x": e[0],
                "title": e[1].toString(),
                "text": `Entry at ${e[1]}`
            }
        });

        let sell = datumElement[0]["observers"]["BuySell"]['sell'].map(e => {
            return {
                "x": e[0],
                "title": e[1].toString(),
                "text": `Stopped at ${e[1]}`
            }
        });

        let indicators = []
        for (var indicator in datumElement[0]["indicators"]) {
            let lines = []
            for (var line in datumElement[0]["indicators"][indicator]) {
                let plotline = {
                    'name': line,
                    'label': indicator,
                    "x": datumElement[0]["indicators"][indicator][line]['values']
                }
                lines.push(plotline)
            }
            indicators.push(lines)
        }

        let data = {
            'buy': buy,
            'sell': sell,
            'indicators': indicators
        };

        

        return <StrategyChart
            tid={this.state.tid}
            provider={this.state.results['provider']}
            symbol={this.state.results['symbol']}
            startDate={this.state.results['start_date']}
            endDate={this.state.results['end_date']}
            data={data}/>;
    }

    getWalkForwardStrategyChart() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const datumElement = this.state.results['optimizations']['WALKFORWARD'];

        if (datumElement.length === 0) {
            return this.getNoDataMsg();
        }

        let buy = [];
        let sell = [];

        for (let i = 0; i < datumElement.length; i++) {
            let buyTmp = datumElement[i]["observers"]["BuySell"]['buy'].map(e => {
                return {
                    "x": e[0],
                    "title": e[1].toString(),
                    "text": `Entry at ${e[1]} (Num Split: ${i})`
                }
            });

            let sellTmp = datumElement[i]["observers"]["BuySell"]['sell'].map(e => {
                return {
                    "x": e[0],
                    "title": e[1].toString(),
                    "text": `Stopped at ${e[1]} (Num Split: ${i})`
                }
            });

            buy = buy.concat(buyTmp);
            sell = sell.concat(sellTmp);
        }

        let data = {
            'buy': buy,
            'sell': sell
        };

        return <StrategyChart
            tid={this.state.tid}
            provider={this.state.results['provider']}
            symbol={this.state.results['symbol']}
            startDate={this.state.results['start_date']}
            endDate={this.state.results['end_date']} data={data}/>;
    }

    getBacktestPnLChart() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const datumElement = this.state.results['optimizations']['BACKTESTING'];

        if (datumElement.length === 0) {
            return this.getNoDataMsg();
        }

        return <PnLChart
            tid={this.state.results['tid']}
            data={datumElement[0]["observers"]["Trades"]['pnl']}/>;
    }

    getWalkForwardPnLChart() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const datumElement = this.state.results['optimizations']['WALKFORWARD'];

        if (datumElement.length === 0) {
            return this.getNoDataMsg();
        }

        let pnl = [];

        for (let i = 0; i < datumElement.length; i++) {
            pnl = pnl.concat(datumElement[i]["observers"]["Trades"]['pnl']);
        }


        return <PnLChart tid={this.state.results['tid']} data={pnl}/>;
    }

    getParametersDistribution() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const optimizations = this.state.results["optimizations"];

        if (optimizations["BACKTESTING"].length === 0 ||
            optimizations["WALKFORWARD"].length === 0 ||
            Object.keys(this.state.results["parameters"]).length === 0) {
            return <p>Current Strategy does not have any custom parameters.</p>
        }

        return (
            <Grid container>
                {Object.keys(this.state.results["parameters"]).map(p=>
                    <Grid item xs={4} key={p}>
                        {this.getParametersDistributionPlot(p)}
                    </Grid>)}
            </Grid>
        );
    }

    getParametersDistributionPlot(p) {
        const yData = [];
        const chartTitle = p.charAt(0).toUpperCase() + p.slice(1);
        const xLabel = 'Out of Sample';
        const yLabel = 'Parameter';
        const vLineLabel = 'Backtesting';
        const vLineValue = this.state.results["optimizations"]["BACKTESTING"][0]["parameters"][p];


        (this.state.results["optimizations"]["WALKFORWARD"]
            .sort(function (a, b) {
                return a["num_split"] < b["num_split"] ? -1 :  a["num_split"] > b["num_split"] ? 1 : 0;
            })
            .forEach(r => yData.push(r["parameters"][p])));

        return <ParametersDistributionPlot
            tid={this.state.results['tid']}
            chartTitle={chartTitle}
            xLabel={xLabel}
            yLabel={yLabel}
            vLineLabel={vLineLabel}
            vLineValue={vLineValue}
            data={yData}/>;
    }

    getHeatMapChart() {
        if (Object.keys(this.state.results).length === 0) {
            return this.getNoDataMsg();
        }

        const optimizations = this.state.results["optimizations"];
        if (optimizations["BACKTESTING"].length === 0 || optimizations["WALKFORWARD"].length === 0) {
            return <p>No Data Available. You need to perform both Backtesting and Walkforward testing
                to get parameters distribution.</p>
        }

        return <HeatMapChart tid={this.state.results['tid']}/>;
    }

    handleInputChange(event){
        this.setState({tid: event.target.value}, this.fetchResults);
    }

    fetchResults() {
        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${this.state.tid}`)
            .then(response => response.json())
            .then(data => this.setState({results: data}))
            .catch(error => console.log(error));
    }

}

export default withStyles(styles)(Evaluation);
