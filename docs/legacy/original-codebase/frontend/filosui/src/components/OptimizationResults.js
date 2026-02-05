import React from 'react';
import 'fontsource-roboto';
import {withStyles} from '@material-ui/core/styles';
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import Title from "./Title";
import {createMuiTheme} from "@material-ui/core";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableRow from "@material-ui/core/TableRow";
import TableHead from "@material-ui/core/TableHead";
import Table from "@material-ui/core/Table";
import TableContainer from "@material-ui/core/TableContainer";
import IconButton from '@material-ui/core/IconButton';
import ViewListIcon from '@material-ui/icons/ViewList';
import TreeView from "@material-ui/lab/TreeView";
import TreeItem from "@material-ui/lab/TreeItem";
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import DeleteIcon from '@material-ui/icons/Delete';
import {withRouter} from "react-router-dom";

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
};

class OptimizationResults extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            tableBody: <TableBody/>
        }
    }

    componentDidMount() {
        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results`)
            .then(response => response.json())
            .then(data => {

                data.sort(function (a, b) {
                    return a['creation_time'] < b['creation_time']? 1: a['creation_time'] > b['creation_time']? -1: 0
                });

                let tableBody = (
                    <TableBody>
                        {
                            data.map(row => (
                                <TableRow key={row["tid"]} hover>
                                    <TableCell>{row["test_name"]}</TableCell>
                                    <TableCell>{new Date(row["creation_time"]).toLocaleString()}</TableCell>
                                    <TableCell>{row['strategy']}</TableCell>
                                    <TableCell>{row["provider"]}</TableCell>
                                    <TableCell>{row["symbol"]}</TableCell>
                                    <TableCell>{row["timeframe"]}</TableCell>
                                    <TableCell>{this.getPerformedTests(row["optimizations"])}</TableCell>
                                    <TableCell>
                                        <IconButton onClick={() => this.comparePage(row['tid'])}>
                                            <ViewListIcon/>
                                        </IconButton>
                                        <IconButton onClick={() => this.deleteOptResult(row["tid"])}>
                                            <DeleteIcon/>
                                        </IconButton>
                                    </TableCell>
                                </TableRow>
                            ))
                        }
                    </TableBody>
                );

                this.setState({
                    tableBody: tableBody
                })
            }).catch(error => console.log(error))
    }

    render() {
        const { classes } = this.props;
        return (
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <Title>Results</Title>
                        <TableContainer size="small" >
                            <Table aria-label="Results Table">
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Creation Time</TableCell>
                                        <TableCell>Strategy</TableCell>
                                        <TableCell>Provider</TableCell>
                                        <TableCell>Symbol</TableCell>
                                        <TableCell>Timeframe</TableCell>
                                        <TableCell>Tests</TableCell>
                                        <TableCell/>
                                    </TableRow>
                                </TableHead>
                                {this.state.tableBody}
                            </Table>
                        </TableContainer>
                    </Paper>
                </Grid>
            </Grid>);
    }

    getPerformedTests(rowElement) {
        let tests = Object.keys(rowElement);
        return (
            <TreeView
                defaultCollapseIcon={<ExpandMoreIcon />}
                defaultExpandIcon={<ChevronRightIcon />}>
                {tests.filter(t => rowElement[t].length > 0).map((t, i) => (
                    <TreeItem nodeId={i.toString()} key={t} label={t}>
                        {rowElement[t].map((test, j) =>
                            <TreeItem
                                nodeId={`params-${i}-${j}`}
                                key={`params-${i}-${j}`}
                                label={"Parameters " + (test["num_split"] !== null? test["num_split"]:"")}>
                                {
                                    Object.keys(test['parameters']).map(p =>
                                        <TreeItem
                                            nodeId={`params-${i}-${j}-${p}`}
                                            key={`params-${i}-${j}-${p}`}
                                            label={p + ": " + test['parameters'][p]}
                                        />
                                    )
                                }
                            </TreeItem>)
                        }
                    </TreeItem>
                ))}
            </TreeView>)
    }

    comparePage(tid) {
        this.props.history.push(`/evaluate/${tid}`);
    }

    deleteOptResult(tid) {
        fetch(`${process.env.REACT_APP_REST_API_URL}/optimization/results/${tid}`,
            {method: 'DELETE'})
            .then(res => this.componentDidMount())
    }

}

export default withRouter(withStyles(styles)(OptimizationResults));
