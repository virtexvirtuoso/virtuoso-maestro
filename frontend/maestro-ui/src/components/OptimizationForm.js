import React from 'react';
import 'fontsource-roboto';
import {withStyles} from '@material-ui/core/styles';
import Title from "./Title";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import Grid from "@material-ui/core/Grid";
import Box from "@material-ui/core/Box";
import IconButton from "@material-ui/core/IconButton";
import PlayCircleFilledWhiteIcon from "@material-ui/icons/PlayCircleFilledWhite";
import LinearProgressWithLabel from "./LinearProgressWithLabel";
import TextField from "@material-ui/core/TextField";
import FormLabel from "@material-ui/core/FormLabel";
import RadioGroup from "@material-ui/core/RadioGroup";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Radio from "@material-ui/core/Radio";
import { DateTimePicker, MuiPickersUtilsProvider } from "@material-ui/pickers";
import DateFnsUtils from '@date-io/date-fns';

const styles = {
    formControl: {
        margin: 5,
        minWidth: 120,
    },
    button: {
        margin: 5,
    }
};

class OptimizationForm extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            testName: "Test Name",
            provider: "",
            providers: [],
            symbol: "",
            symbols: [],
            binSize: "1d",
            strategy: "",
            strategies: [],
            optType: "BACKTESTING",
            isRunning: false,
            progress: 0,
            isError: false,
            testNameHelperText: "",
            testNameVariant: "outlined",
            cash: 10000,
            commissions: 0.01,
            strategiesParameters: [],
            startDate: new Date(2000, 1, 1),
            endDate: new Date()
        };

        this.handleInputChange = this.handleInputChange.bind(
            this
        );

        this.submitTest = this.submitTest.bind(
            this
        );
    }

    componentDidMount() {
        fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/available`)
            .then(response => response.json())
            .then(data => {
                this.setState({
                    strategies: data,
                    strategy: data.length > 0 ? data[0] : ""
                }, () => this.updateStrategyParams(this.state.strategy));

            }).catch(e => alert(`Something went wrong: ${e}`))

        fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/available`)
            .then(response => response.json())
            .then(data => {
                this.setState({
                    providers: data,
                    provider: data.length > 0 ? data[0] : ""
                }, () => this.updateSymbolsAvailable(this.state.provider));
            }).catch(e => alert(`Something went wrong: ${e}`))
        if (this.state.provider.length > 0) {
            fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/symbols`)
                .then(response => response.json())
                .then(data => {
                    this.setState({
                        symbols: data,
                        symbol: data.length > 0 ? data[0] : ""
                    });
                }).catch(e => alert(`Something went wrong: ${e}`))
        }

    }

    handleInputChange(event) {
        const target = event.target;
        const value = target.value;
        const name = target.name;
        this.setState({
            [name]: value,
        });

        if (name === "testName") {
            this.setState({
                isError: false,
                testNameHelperText: ""
            })
        }

        if (name === "strategy") {
            this.updateStrategyParams(value);
        }

        if (name === "provider") {
            this.updateSymbolsAvailable(value);
        }

    }

    updateStrategyParams(strategyName) {
        fetch(`${process.env.REACT_APP_REST_API_URL}/strategy/${strategyName}/params`)
            .then(response => response.json())
            .then(data => {
                let newState = {
                    strategiesParameters: Object.keys(data)
                }
                Object.keys(data).forEach(k => newState[k] = data[k]);
                this.setState(newState);
            }).catch(e => alert(`Something went wrong: ${e}`))
    }

    updateSymbolsAvailable(providerName) {
        fetch(`${process.env.REACT_APP_REST_API_URL}/datasource/${providerName}/symbols`)
            .then(response => response.json())
            .then(data => {
                this.setState({
                        symbols: data,
                        symbol: data.length > 0 ? data[0] : ""})
            }).catch(e => alert(`Something went wrong: ${e}`))
    }

    render() {
        const { classes } = this.props;
        return (<React.Fragment >
            <Title>New Test</Title>
            <FormControl className={classes.formControl}>
                <TextField
                    key="test-name-text"
                    error={this.state.isError}
                    id="test-name-text"
                    label="Test Name"
                    name="testName"
                    value={this.state.testName}
                    helperText={this.state.testNameHelperText}
                    variant={this.state.testNameVariant}
                    onChange={this.handleInputChange}
                />
            </FormControl>
            <FormControl className={classes.formControl}>
                <MuiPickersUtilsProvider utils={DateFnsUtils}>
                    <DateTimePicker
                        label="From"
                        name="startDate"
                        value={this.state.startDate}
                        onChange={d => this.setState({startDate: d})}
                        autoOk
                        ampm={false}
                        format="yyyy/MM/dd HH:mm"
                        disableFuture
                        showTodayButton
                    />
                </MuiPickersUtilsProvider>
            </FormControl>
            <FormControl className={classes.formControl}>
                <MuiPickersUtilsProvider utils={DateFnsUtils}>
                    <DateTimePicker
                        label="To"
                        // inputVariant="outlined"
                        name="endDate"
                        value={this.state.endDate}
                        onChange={d => this.setState({endDate: d})}
                        autoOk
                        format="yyyy/MM/dd HH:mm"
                        ampm={false}
                        disableFuture
                        showTodayButton
                    />
                </MuiPickersUtilsProvider>
            </FormControl>
            <FormControl className={classes.formControl}>
                <InputLabel id="provider-select-label">Provider</InputLabel>
                <Select
                    labelId="provider-select-label"
                    id="provider-select"
                    name = "provider"
                    value={this.state.provider}
                    onChange={this.handleInputChange}
                >
                    {this.state.providers.map(p => <MenuItem key={p} value={p}>{p}</MenuItem>)}

                </Select>
            </FormControl>
            <FormControl className={classes.formControl}>
                <InputLabel id="symbol-select-label">Symbol</InputLabel>
                <Select
                    labelId="symbol-select-label"
                    id="symbol-select"
                    name = "symbol"
                    value={this.state.symbol}
                    onChange={this.handleInputChange}
                >
                    {this.state.symbols.map(p => <MenuItem key={p} value={p}>{p}</MenuItem>)}
                </Select>
            </FormControl>
            <FormControl className={classes.formControl} >
                <InputLabel id="binsize-select-label">Time Frame</InputLabel>
                <Select
                    labelId="binsize-select-label"
                    id="binsize-select"
                    name = "binSize"
                    value={this.state.binSize}
                    onChange={this.handleInputChange}
                >
                    <MenuItem key="1d"  value={"1d"}>1 Day</MenuItem>
                    <MenuItem key={"1h"} value={"1h"}>1 Hour</MenuItem>
                    <MenuItem key={"5m"} value={"5m"}>5 Minutes</MenuItem>
                    <MenuItem key={"1m"} value={"1m"}>1 Minutes</MenuItem>
                </Select>
            </FormControl>
            <FormControl className={classes.formControl}>
                <TextField
                    key="cash-text"
                    error={this.state.isError}
                    id="cash-text"
                    label="Cash"
                    name="cash"
                    defaultValue={this.state.cash}
                    onChange={this.handleInputChange}
                />
            </FormControl>
            <FormControl className={classes.formControl}>
                <TextField
                    key="commissions-text"
                    error={this.state.isError}
                    id="commissions-text"
                    label="Commissions"
                    name="commissions"
                    defaultValue={this.state.commissions}
                    onChange={this.handleInputChange}
                />
            </FormControl>
            <FormControl className={classes.formControl} >
                <InputLabel id="strategy-select-label">Strategy</InputLabel>
                <Select
                    labelId="strategy-select-label"
                    id="strategy-select"
                    name="strategy"
                    value={this.state.strategy}
                    onChange={this.handleInputChange}
                >
                    {
                        this.state.strategies.map(row => <MenuItem key={row} value={row}>{row}</MenuItem>)
                    }
                </Select>
            </FormControl>
            <FormControl className={classes.formControl} >
                {
                    this.state.strategiesParameters.map(row => this.generateStrategyParamsInput(row))
                }
            </FormControl>
            <FormControl  className={classes.formControl}>
                <FormLabel>Optimization Type</FormLabel>
                <RadioGroup
                    id="optType"
                    name="optType" value={this.state.optType} onChange={this.handleInputChange}>
                    <FormControlLabel value="BACKTESTING" control={<Radio />} label="Backtesting" />
                    <FormControlLabel value="WALKFORWARD" control={<Radio />} label="Walk Forward" />
                    <FormControlLabel value="BOTH" control={<Radio />} label="Backtesting + Walk Forward" />
                </RadioGroup>
            </FormControl>
            <Grid
                container
                direction="row"
                justify="flex-end"
                alignItems="center"
            >
                <Box>
                    <IconButton aria-label="Run new test"
                                onClick={this.submitTest}
                                className={classes.button}>
                        <PlayCircleFilledWhiteIcon color="primary" fontSize="large" />
                    </IconButton>
                </Box>
            </Grid>
            <Grid hidden={!this.state.isRunning} >
                <LinearProgressWithLabel value={this.state.progress} />
            </Grid>
        </React.Fragment>);
    }

    submitTest() {

        let params = {
            "test_name": this.state.testName,
            "symbol": this.state.symbol,
            "provider": this.state.provider,
            "bin_size": this.state.binSize,
            "strategy": this.state.strategy,
            "kind": this.state.optType,
            "cash": this.state.cash,
            "commissions": this.state.commissions,
            "start_date": this.state.startDate.getTime(),
            "end_date": this.state.endDate.getTime(),
            "strategy_params": {}
        }

        this.state.strategiesParameters.forEach(p => {
            params["strategy_params"][p] = this.state[p]
        });

        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        };

        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/new/`, requestOptions)
            .then(response => {
                const statusCode = response.status;
                const data = response.json()
                return Promise.all([statusCode, data])
            }).then(([status_code, data]) => {
            if (status_code !== 200) {
                this.setState({
                    isError: true,
                    testNameHelperText: data['error'],
                    isRunning: false
                })
            } else {
                this.setState({
                    isRunning: true,
                    progress: 0
                })

                const timer = setInterval(() => {
                    if (this.state.progress === 100){
                        clearInterval(timer);
                    } else {
                        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/progress/${data['tid']}`)
                            .then(response => Promise.all([response.status, response.json()]))
                            .then(([status_code, data]) => {

                                let progress = 0;
                                if (Object.keys(data).length > 0) {
                                    let current = 0, total = 0;

                                    for(let optType of ['BACKTESTING', 'WALKFORWARD']){
                                        let cur_test = data['optimizations'][optType];

                                        if (cur_test !== undefined &&
                                            (this.state.optType === optType ||
                                                this.state.optType === "BOTH" ||
                                                cur_test['current'] < cur_test['total'])) {
                                            current += cur_test['current']
                                            total += cur_test['total']
                                        }
                                    }
                                    progress = current / total * 100
                                }

                                this.setState({
                                    progress: progress
                                })

                            }).catch(e => alert(`Something went wrong: ${e}`))
                    }
                }, 800);
                return () => {
                    clearInterval(timer);
                };
            }
        }).catch(e => alert(`Something went wrong: ${e}`));

    }

    generateStrategyParamsInput(row) {
        const { classes } = this.props;
        return (
            <FormControl key={row + "-form-control"} className={classes.formControl}>
                <TextField
                    key={row}
                    id={row + "-text"}
                    label={row}
                    name={row}
                    type="number"
                    value={this.state[row]}
                    onChange={this.handleInputChange}
                />
            </FormControl>)
    }
}

export default withStyles(styles)(OptimizationForm);
