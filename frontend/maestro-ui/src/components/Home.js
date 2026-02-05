import React from 'react';
import 'fontsource-roboto';
import {withStyles} from '@material-ui/core/styles';
import Grid from "@material-ui/core/Grid";
import Paper from "@material-ui/core/Paper";
import CandleStickChart from "./CandleStickChart";
import OptimizationForm from "./OptimizationForm";
import {createMuiTheme} from "@material-ui/core";

const theme = createMuiTheme();

const styles = {
    paper: {
        padding: theme.spacing(2),
        display: 'flex',
        overflow: 'auto',
        flexDirection: 'column',
    },
};

class Home extends React.Component {


    render() {
        const { classes } = this.props;
        return (
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <CandleStickChart/>
                    </Paper>
                </Grid>
                <Grid item xs={12}>
                    <Paper className={classes.paper}>
                        <OptimizationForm/>
                    </Paper>
                </Grid>
            </Grid>);
    }

}

export default withStyles(styles)(Home);
